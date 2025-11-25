# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib 
from model import TransformerNBEATS 

# ================= 配置区域 =================
# 1. 路径配置
BASE_DIR = Path("outputs/clean")
DATA_FILE = BASE_DIR / "dataset_ready_for_research.csv"
MODEL_SAVE_PATH = "transformer_best.pth"
SCALER_SAVE_PATH = "scaler.pkl"

# 2. 模型超参数
SEQ_LEN = 96       # 历史输入长度 (24h * 4 = 96 points)
PRED_LEN = 96      # 预测输出长度 (24h * 4 = 96 points)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. 特征配置 (根据您上传的 CSV 修改)
# 必须与 CSV 中的列名完全一致
# 顺序很重要：Target_Power 必须放在第一个，作为预测目标 (Index 0)
FEATURE_COLS = [
    'Target_Power',  # 目标变量 (必须在第0位)
    'NWP_GHI',       # 总辐射
    'NWP_DNI',       # 直射辐射 (新增)
    'NWP_DHI',       # 散射辐射 (新增)
    'NWP_Temp',      # 气温
    'NWP_Wind',      # 风速
    'NWP_Humidity',  # 湿度 (新增)
    'NWP_Cloud',     # 云量 (新增)
    'NWP_Precip'     # 降水 (新增)
]

# 自动计算输入维度 (1个历史目标 + 8个天气特征 = 9)
INPUT_DIM = len(FEATURE_COLS)
# ===========================================

class PVDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.x = torch.FloatTensor(data_x)
        self.y = torch.FloatTensor(data_y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def create_sequences(df_vals, seq_len, pred_len):
    """
    构造滑动窗口样本
    X: [t-seq_len : t] 的所有特征
    Y: [t : t+pred_len] 的 实际功率 (Target_Power)
    """
    xs, ys = [], []
    total_len = len(df_vals)
    
    for i in range(total_len - seq_len - pred_len):
        hist_start = i
        hist_end = i + seq_len
        pred_start = hist_end
        pred_end = hist_end + pred_len
        
        # X: 取这段时间窗口内的"所有特征" (包括功率和天气)
        # 形状: [seq_len, INPUT_DIM]
        x_seq = df_vals[hist_start:hist_end, :] 
        
        # Y: 只取这段时间窗口内的"Target_Power" (第0列)
        # 形状: [pred_len]
        y_seq = df_vals[pred_start:pred_end, 0] 
        
        xs.append(x_seq)
        ys.append(y_seq)
        
    return np.array(xs), np.array(ys)

def train_process():
    print(f"--- [Step 3] 训练 Transformer 模型 (修正特征版) ---")
    print(f"--- 设备: {DEVICE} ---")
    
    # 1. 读取数据
    if not DATA_FILE.exists():
        print(f"错误: 找不到数据集 {DATA_FILE}")
        return
    
    # index_col=0 会自动处理 'Unnamed: 0' 这种索引列
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    print(f"原始数据加载成功，形状: {df.shape}")
    
    # 2. 特征选择与归一化
    # 检查列是否存在
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        print(f"错误: CSV中缺少以下列: {missing_cols}")
        print(f"CSV现有列: {df.columns.tolist()}")
        return

    # 只选取配置好的列，并保证顺序
    data = df[FEATURE_COLS].values
    print(f"使用特征 ({INPUT_DIM}维): {FEATURE_COLS}")
    
    # 归一化 (MinMax 到 0~1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 保存归一化器
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"归一化器已保存至: {SCALER_SAVE_PATH}")
    
    # 3. 构造数据集
    print("构造样本中...")
    # 传入 numpy array 加速
    X, Y = create_sequences(data_scaled, SEQ_LEN, PRED_LEN)
    print(f"样本构造完成 - X: {X.shape}, Y: {Y.shape}")
    
    if len(X) == 0:
        print("错误: 样本数量为0，请检查 SEQ_LEN/PRED_LEN 是否超过了数据总长度。")
        return

    # 划分 训练集/验证集 (80% / 20%)
    train_size = int(len(X) * 0.8)
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_val, Y_val = X[train_size:], Y[train_size:]
    
    train_loader = DataLoader(PVDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PVDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. 初始化模型
    model = TransformerNBEATS(
        num_stacks=2,
        num_blocks_per_stack=2,
        input_dim=INPUT_DIM,     # 自动更新为 9
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=128,
        input_seq_len=SEQ_LEN,
        output_len=PRED_LEN,
        dropout=0.1
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. 训练循环
    best_val_loss = float('inf')
    
    print("\n开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print(f"\n训练结束！最佳模型已保存至: {MODEL_SAVE_PATH}")
    print(f"Best Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    train_process()