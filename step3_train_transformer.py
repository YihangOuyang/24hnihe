# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib 
import pvlib 
import warnings

# 确保 model.py 在同一目录下
try:
    from step0_model import TransformerNBEATS 
except ImportError:
    print("提示: 请确保 model.py 包含 TransformerNBEATS 类")

warnings.filterwarnings("ignore")

# ================= 配置区域 =================
CURRENT_DIR = Path(__file__).parent.absolute()
BASE_DIR = CURRENT_DIR / "outputs" / "clean"
DATA_FILE = BASE_DIR / "dataset_ready_for_research_15min.csv"

MODEL_SAVE_PATH = "transformer_best.pth"
SCALER_SAVE_PATH = "scaler.pkl"
LOSS_PLOT_PATH = "training_loss_curve.png"

SEQ_LEN = 96
PRED_LEN = 96
BATCH_SIZE = 32
EPOCHS = 100 
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 15 

LATITUDE = 34.05
LONGITUDE = -118.24
ALTITUDE = 71
TIMEZONE = "UTC"

RAW_FEATURE_COLS = [
    'Target_Power', 
    'NWP_GHI', 'NWP_DNI', 'NWP_DHI', 
    'NWP_Temp', 'NWP_Wind', 'NWP_Humidity', 'NWP_Cloud', 'NWP_Precip'
]
# ===========================================

class PVDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.x = torch.FloatTensor(data_x)
        self.y = torch.FloatTensor(data_y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def add_solar_features(df):
    """生成太阳几何特征"""
    print("   [预处理] 正在生成太阳几何特征...")
    try:
        times = pd.to_datetime(df.index, utc=True)
    except:
        times = pd.to_datetime(df.index).tz_localize('UTC')
    
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz=TIMEZONE)
    solpos = loc.get_solarposition(times)
    
    df['Solar_El_sin'] = np.sin(np.radians(solpos['elevation']))
    df['Solar_Az_sin'] = np.sin(np.radians(solpos['azimuth']))
    df['Solar_Az_cos'] = np.cos(np.radians(solpos['azimuth']))
    return df

def create_sequences(data_array, seq_len, pred_len):
    """
    构造滑动窗口样本
    data_array: 归一化后的 numpy 数组 (N, Features)
    假设第0列是 Target_Power
    """
    xs, ys = [], []
    total_len = len(data_array)
    for i in range(total_len - seq_len - pred_len + 1):
        x_seq = data_array[i : i + seq_len, :] 
        y_seq = data_array[i + seq_len : i + seq_len + pred_len, 0] # 只预测功率(第0列)
        xs.append(x_seq)
        ys.append(y_seq)
    return np.array(xs), np.array(ys)

def train_process():
    print(f"--- [Step 3] 训练 Transformer (严格防泄露版) ---")
    print(f"--- 设备: {DEVICE} ---")
    
    if not DATA_FILE.exists():
        print(f"❌ 错误: 找不到数据集 {DATA_FILE}")
        return
    
    # 1. 读取与特征工程
    df = pd.read_csv(DATA_FILE, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df = add_solar_features(df)
    
    SOLAR_COLS = ['Solar_El_sin', 'Solar_Az_sin', 'Solar_Az_cos']
    FULL_COLS = RAW_FEATURE_COLS + SOLAR_COLS
    print(f"   特征维度: {len(FULL_COLS)}")
    
    # 2. 提取数据并处理 NaN
    data_raw = df[FULL_COLS].values
    data_raw = np.nan_to_num(data_raw, nan=0.0)
    
    # =========================================================
    # 【核心修正 1】先划分数据，再归一化 (Prevent Leakage)
    # =========================================================
    total_len = len(data_raw)
    train_split_idx = int(total_len * 0.70)
    val_split_idx = int(total_len * 0.85)
    
    # 切分原始数据
    train_data_raw = data_raw[:train_split_idx]
    val_data_raw   = data_raw[train_split_idx:val_split_idx]
    test_data_raw  = data_raw[val_split_idx:]
    
    print(f"   数据划分: Train={len(train_data_raw)}, Val={len(val_data_raw)}, Test={len(test_data_raw)}")

    # 仅在训练集上 Fit Scaler
    print("   [预处理] 执行 fit_transform (仅基于训练集)...")
    scaler = MinMaxScaler()
    scaler.fit(train_data_raw) 
    
    # 保存 Scaler (重要！后续推理需要用到完全相同的参数)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    
    # 分别 Transform
    train_data_scaled = scaler.transform(train_data_raw)
    val_data_scaled   = scaler.transform(val_data_raw)
    test_data_scaled  = scaler.transform(test_data_raw)
    
    # =========================================================
    # 3. 构造序列 (Sliding Window)
    # =========================================================
    X_train, Y_train = create_sequences(train_data_scaled, SEQ_LEN, PRED_LEN)
    X_val, Y_val     = create_sequences(val_data_scaled,   SEQ_LEN, PRED_LEN)
    X_test, Y_test   = create_sequences(test_data_scaled,  SEQ_LEN, PRED_LEN)
    
    # DataLoader
    train_loader = DataLoader(PVDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(PVDataset(X_val, Y_val),   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(PVDataset(X_test, Y_test),  batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. 初始化模型
    input_dim = len(FULL_COLS)
    model = TransformerNBEATS(
        num_stacks=1, num_blocks_per_stack=1, input_dim=input_dim, 
        d_model=32, nhead=4, num_encoder_layers=2, 
        dim_feedforward=64, input_seq_len=SEQ_LEN, output_len=PRED_LEN, dropout=0.3
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # 5. 训练循环
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0 
    
    print("\n开始训练...")
    for epoch in range(EPOCHS):
        # Training
        model.train()
        batch_train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_x)
            
            # 【修正形状】确保维度对齐
            if output.dim() == 3 and output.shape[-1] == 1:
                output = output.squeeze(-1)
            
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
        
        epoch_train_loss = np.mean(batch_train_losses)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                output = model(batch_x)
                if output.dim() == 3 and output.shape[-1] == 1:
                    output = output.squeeze(-1)
                loss = criterion(output, batch_y)
                batch_val_losses.append(loss.item())
        
        epoch_val_loss = np.mean(batch_val_losses)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train: {epoch_train_loss:.6f} | Val: {epoch_val_loss:.6f}", end="")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0 
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(" -> Saved (*)")
        else:
            patience_counter += 1
            print(f" -> Patience {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print(f"\n[早停] Val Loss 连续 {PATIENCE} 轮未下降。")
            break
            
    # =========================================================
    # 【核心修正 2】反归一化评估 (Physical Meaning)
    # =========================================================
    print("\n--- [Final Evaluation] 测试集最终评估 (真实物理单位) ---")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    
    test_preds = []
    test_trues = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            output = model(batch_x)
            if output.dim() == 3: output = output.squeeze(-1)
            
            test_preds.append(output.cpu().numpy())
            test_trues.append(batch_y.numpy())
            
    y_pred_scaled = np.concatenate(test_preds, axis=0)
    y_true_scaled = np.concatenate(test_trues, axis=0)
    
    # 辅助函数：利用 scaler 反归一化只包含 Power 的那一列
    def inverse_transform_target(y_flat, scaler, feature_dim):
        # 构造一个形状为 [N, feature_dim] 的临时矩阵
        # 假设 Target_Power 是第 0 列 (与 RAW_FEATURE_COLS 顺序一致)
        dummy = np.zeros((len(y_flat), feature_dim))
        dummy[:, 0] = y_flat 
        res = scaler.inverse_transform(dummy)
        return res[:, 0] # 返回第 0 列（真实功率）

    # 展平数据以便反归一化
    # y_pred_scaled shape: [N_samples, PRED_LEN] -> flatten
    y_pred_real = inverse_transform_target(y_pred_scaled.flatten(), scaler, input_dim)
    y_true_real = inverse_transform_target(y_true_scaled.flatten(), scaler, input_dim)
    
    # 物理截断 (功率 >= 0)
    y_pred_real = np.maximum(y_pred_real, 0)
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    mae = mean_absolute_error(y_true_real, y_pred_real)
    
    print(f"Test Set Results (Real Power):")
    print(f" >> RMSE : {rmse:.4f} kW")
    print(f" >> MAE  : {mae:.4f} kW")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training Loss (Scaled Space)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(LOSS_PLOT_PATH)
    print(f"Loss 曲线已保存: {LOSS_PLOT_PATH}")

if __name__ == "__main__":
    train_process()