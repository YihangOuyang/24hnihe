# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import joblib
from pathlib import Path
import time

# 引入你的 Transformer 模型定义
from model import TransformerNBEATS 

# ================= 配置区域 =================
BASE_DIR = Path("outputs/clean")
DATA_FILE = BASE_DIR / "dataset_ready_for_research.csv"
TRANSFORMER_PATH =  "transformer_best.pth" # 预训练好的 Transformer
SCALER_PATH =  "scaler.pkl"

SEQ_LEN = 96
PRED_LEN = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P_RATED = 30.1
BATCH_SIZE = 64
# ===========================================

# --- 1. 定义深度学习基准模型 (LSTM & MLP) ---

class LSTMModel(nn.Module):
    """LSTM 基准模型: 代表上一代时序预测 SOTA"""
    def __init__(self, input_dim, hidden_dim, output_len, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_len)
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        # LSTM 输出: out, (h_n, c_n)
        # 我们只取最后一个时间步的隐状态用于预测
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :] 
        return self.fc(last_hidden)

class MLPModel(nn.Module):
    """MLP 基准模型: 证明结构(Attention)比单纯的深度(Deep)更重要"""
    def __init__(self, input_len, input_dim, output_len):
        super(MLPModel, self).__init__()
        self.flatten_dim = input_len * input_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_len)
        )
        
    def forward(self, x):
        return self.net(x)

# --- 2. 辅助函数 ---

def load_and_prep_data():
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    cols = ['Target_Power', 'NWP_GHI', 'NWP_DNI', 'NWP_DHI', 
            'NWP_Temp', 'NWP_Wind', 'NWP_Humidity', 'NWP_Cloud', 'NWP_Precip']
    # 确保列存在
    valid_cols = [c for c in cols if c in df.columns]
    data = df[valid_cols].values
    
    scaler = joblib.load(SCALER_PATH)
    data_scaled = scaler.transform(data)
    
    xs, ys = [], []
    for i in range(len(data) - SEQ_LEN - PRED_LEN):
        xs.append(data_scaled[i:i+SEQ_LEN])
        ys.append(data_scaled[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN, 0]) # Target Power
        
    X = np.array(xs)
    Y = np.array(ys)
    
    # 划分训练/测试 (80/20)
    split = int(len(X) * 0.8)
    return X[:split], Y[:split], X[split:], Y[split:], scaler, len(valid_cols)

def train_torch_model(model, X_train, Y_train, epochs=10):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"   Training {model.__class__.__name__}...")
    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    return model

def predict_torch(model, X_test):
    model.eval()
    preds = []
    dataset = TensorDataset(torch.FloatTensor(X_test))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for bx in loader:
            bx = bx[0].to(DEVICE)
            out = model(bx)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)

def inverse_transform_y(y_scaled, scaler, input_dim):
    # 构建 dummy 矩阵以进行反归一化
    dummy = np.zeros((y_scaled.size, input_dim))
    dummy[:, 0] = y_scaled.flatten()
    inv = scaler.inverse_transform(dummy)[:, 0]
    return inv.reshape(y_scaled.shape)

def calc_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"   >> {name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
    return {'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# --- 主程序 ---

def run_full_comparison():
    print("--- [Step 5] 全面模型对比实验 (Full Benchmark) ---")
    
    # 1. 数据加载
    X_train, Y_train, X_test, Y_test, scaler, input_dim = load_and_prep_data()
    
    # 反归一化真实标签
    Y_true = inverse_transform_y(Y_test, scaler, input_dim)
    print(f"测试集样本: {len(Y_test)}, 输入特征数: {input_dim}")
    
    results = []
    predictions = {} # 存储所有模型的预测结果用于绘图
    
    # ==========================================
    # Model 1: Persistence (智能持续法)
    # ==========================================
    print("\n1. Running Persistence...")
    # 假设明天此时功率 = 今天此时功率 (输入序列的最后96个点)
    # X_test shape: [N, 96, F], index 0 is Power
    Y_pred_pers_scaled = X_test[:, :, 0] 
    Y_pred_pers = inverse_transform_y(Y_pred_pers_scaled, scaler, input_dim)
    results.append(calc_metrics(Y_true.flatten(), Y_pred_pers.flatten(), "Persistence"))
    predictions['Persistence'] = Y_pred_pers
    
    # ==========================================
    # Model 2: XGBoost (传统ML最强)
    # ==========================================
    print("\n2. Training XGBoost (这可能需要一分钟)...")
    # XGBoost 需要 2D 输入 [N, Seq*Feat]
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # 使用 MultiOutputRegressor 包装 XGBoost
    # 仅使用部分数据加速演示，正式跑建议用更多数据
    xgb_model = MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1
    ))
    xgb_model.fit(X_train_flat[:5000], Y_train[:5000]) # 限制样本数加速
    
    Y_pred_xgb_scaled = xgb_model.predict(X_test_flat)
    Y_pred_xgb = inverse_transform_y(Y_pred_xgb_scaled, scaler, input_dim)
    results.append(calc_metrics(Y_true.flatten(), Y_pred_xgb.flatten(), "XGBoost"))
    predictions['XGBoost'] = Y_pred_xgb

    # ==========================================
    # Model 3: MLP (简单深度网络)
    # ==========================================
    print("\n3. Training MLP...")
    mlp = MLPModel(SEQ_LEN, input_dim, PRED_LEN)
    mlp = train_torch_model(mlp, X_train, Y_train, epochs=15)
    Y_pred_mlp_scaled = predict_torch(mlp, X_test)
    Y_pred_mlp = inverse_transform_y(Y_pred_mlp_scaled, scaler, input_dim)
    results.append(calc_metrics(Y_true.flatten(), Y_pred_mlp.flatten(), "MLP"))
    predictions['MLP'] = Y_pred_mlp

    # ==========================================
    # Model 4: LSTM (RNN代表)
    # ==========================================
    print("\n4. Training LSTM...")
    lstm = LSTMModel(input_dim, 64, PRED_LEN, num_layers=2)
    lstm = train_torch_model(lstm, X_train, Y_train, epochs=15)
    Y_pred_lstm_scaled = predict_torch(lstm, X_test)
    Y_pred_lstm = inverse_transform_y(Y_pred_lstm_scaled, scaler, input_dim)
    results.append(calc_metrics(Y_true.flatten(), Y_pred_lstm.flatten(), "LSTM"))
    predictions['LSTM'] = Y_pred_lstm

    # ==========================================
    # Model 5: Transformer (Ours)
    # ==========================================
    print("\n5. Loading Transformer (Ours)...")
    # 初始化模型结构
    transformer = TransformerNBEATS(
        num_stacks=2, num_blocks_per_stack=2, input_dim=input_dim, 
        d_model=64, nhead=4, num_encoder_layers=2, 
        dim_feedforward=128, input_seq_len=SEQ_LEN, output_len=PRED_LEN
    ).to(DEVICE)
    
    if TRANS_PATH.exists():
        transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=DEVICE))
    else:
        print("警告: 找不到预训练的 Transformer 模型，将使用随机权重进行演示！")
    
    Y_pred_trans_scaled = predict_torch(transformer, X_test)
    Y_pred_trans = inverse_transform_y(Y_pred_trans_scaled, scaler, input_dim)
    results.append(calc_metrics(Y_true.flatten(), Y_pred_trans.flatten(), "Transformer (Ours)"))
    predictions['Transformer'] = Y_pred_trans

    # --- 结果汇总与保存 ---
    df_res = pd.DataFrame(results)
    df_res.to_csv(BASE_DIR / "final_benchmark_results.csv", index=False)
    print("\n=== 最终对比结果 ===")
    print(df_res)
    
    # --- 绘图 ---
    print("\n绘制对比图...")
    # 随机选一个样本
    idx = np.random.randint(0, len(Y_test))
    
    plt.figure(figsize=(14, 7))
    plt.plot(Y_true[idx], 'k-', linewidth=3, label='Ground Truth')
    plt.plot(predictions['Persistence'][idx], color='gray', linestyle='--', alpha=0.5, label='Persistence')
    plt.plot(predictions['XGBoost'][idx], color='green', linestyle=':', label='XGBoost')
    plt.plot(predictions['LSTM'][idx], color='blue', linestyle='-.', label='LSTM')
    plt.plot(predictions['Transformer'][idx], color='red', linewidth=2.5, label='Transformer (Ours)')
    
    plt.title(f"Model Comparison: 24h Forecast (Sample #{idx})")
    plt.xlabel("Time Steps (15min)")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(BASE_DIR / "full_model_comparison.png")
    print(f"对比图已保存: {BASE_DIR / 'full_model_comparison.png'}")

if __name__ == "__main__":
    TRANS_PATH = Path(TRANSFORMER_PATH)
    run_full_comparison()