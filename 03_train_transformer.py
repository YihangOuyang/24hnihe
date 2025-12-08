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
from model import TransformerNBEATS 

warnings.filterwarnings("ignore")

# ================= 配置区域 =================
# 1. 路径配置 (使用绝对路径，更稳健)
CURRENT_DIR = Path(__file__).parent.absolute()
BASE_DIR = CURRENT_DIR / "outputs" / "clean"

# 【修改点 1】文件名修正
# 指向 01 脚本生成的 15分钟分辨率 文件
DATA_FILE = BASE_DIR / "dataset_ready_for_research_15min.csv"

MODEL_SAVE_PATH = "transformer_best.pth"
SCALER_SAVE_PATH = "scaler.pkl"
LOSS_PLOT_PATH = "training_loss_curve.png"

# 2. 模型超参数
# 15min分辨率: 96点 = 24小时 (日前预测标准)
SEQ_LEN = 96
PRED_LEN = 96
BATCH_SIZE = 32
EPOCHS = 100 
LEARNING_RATE = 1e-4 # 稍微调大一点初始学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 15 

# 3. 地理位置 (Los Angeles)
LATITUDE = 34.05
LONGITUDE = -118.24
ALTITUDE = 71

# 【修改点 2】时区修正
# 必须设为 UTC，因为输入数据已经是 UTC。
# 如果设为 GMT+8，pvlib 会算出错误的太阳位置（相位偏差）。
TIMEZONE = "UTC"

# 4. 特征配置
# 这些列名对应 01 脚本输出的列名
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
    """
    计算太阳几何特征。
    由于 TIMEZONE="UTC"，pvlib 会根据 UTC 时间和洛杉矶经纬度
    计算出正确的当地太阳高度角。
    """
    print("   [预处理] 正在生成太阳几何特征 (Solar Elevation/Azimuth)...")
    
    # 确保索引是 DatetimeIndex 且为 UTC
    # 01脚本保存时已经包含时区信息，这里读取时会自动识别，或者强制指定
    try:
        times = pd.to_datetime(df.index, utc=True)
    except:
        times = pd.to_datetime(df.index).tz_localize('UTC')
    
    # 初始化 Location (tz=UTC)
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz=TIMEZONE)
    solpos = loc.get_solarposition(times)
    
    # 添加特征 (使用 sin/cos 编码避免周期性断裂)
    df['Solar_El_sin'] = np.sin(np.radians(solpos['elevation']))
    df['Solar_Az_sin'] = np.sin(np.radians(solpos['azimuth']))
    df['Solar_Az_cos'] = np.cos(np.radians(solpos['azimuth']))
    return df

def create_sequences(df_vals, seq_len, pred_len):
    """
    构造滑动窗口样本
    假设第0列是 Target_Power
    """
    xs, ys = [], []
    total_len = len(df_vals)
    for i in range(total_len - seq_len - pred_len):
        x_seq = df_vals[i : i + seq_len, :] 
        y_seq = df_vals[i + seq_len : i + seq_len + pred_len, 0] 
        xs.append(x_seq)
        ys.append(y_seq)
    return np.array(xs), np.array(ys)

def train_process():
    print(f"--- [Step 3] 训练 Transformer (Rigorous 70/15/15 Split) ---")
    print(f"--- 设备: {DEVICE} ---")
    print(f"--- 数据源: {DATA_FILE} ---")
    
    if not DATA_FILE.exists():
        print(f"❌ 错误: 找不到数据集 {DATA_FILE}")
        return
    
    # 读取数据
    df = pd.read_csv(DATA_FILE, index_col=0)
    # 强制解析索引为时间格式
    df.index = pd.to_datetime(df.index, utc=True)
    
    # 生成太阳特征
    df = add_solar_features(df)
    
    SOLAR_COLS = ['Solar_El_sin', 'Solar_Az_sin', 'Solar_Az_cos']
    
    # 检查列是否存在
    missing_cols = [c for c in RAW_FEATURE_COLS if c not in df.columns]
    if missing_cols:
        print(f"❌ 错误: 数据集中缺少以下列: {missing_cols}")
        print(f"   当前列: {df.columns.tolist()}")
        return

    FULL_COLS = RAW_FEATURE_COLS + SOLAR_COLS
    INPUT_DIM = len(FULL_COLS)
    print(f"   特征维度: {INPUT_DIM} {FULL_COLS}")
    
    # --- 数据归一化 ---
    data = df[FULL_COLS].values
    # 填充可能的 NaN (物理计算偶尔产生的边缘值)
    data = np.nan_to_num(data, nan=0.0)
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    
    # --- 构造序列 ---
    X, Y = create_sequences(data_scaled, SEQ_LEN, PRED_LEN)
    total_samples = len(X)
    
    # --- 70:15:15 严格划分 ---
    train_split = int(total_samples * 0.70)
    val_split = int(total_samples * 0.85) 
    
    X_train, Y_train = X[:train_split], Y[:train_split]
    X_val, Y_val = X[train_split:val_split], Y[train_split:val_split]
    X_test, Y_test = X[val_split:], Y[val_split:]
    
    print(f"\n数据集划分完成:")
    print(f"   Train Set: {len(X_train)} samples")
    print(f"   Val Set:   {len(X_val)} samples")
    print(f"   Test Set:  {len(X_test)} samples")
    
    train_loader = DataLoader(PVDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PVDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(PVDataset(X_test, Y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 初始化模型 ---
    model = TransformerNBEATS(
        num_stacks=1, num_blocks_per_stack=1, input_dim=INPUT_DIM, 
        d_model=32, nhead=4, num_encoder_layers=2, 
        dim_feedforward=128, input_seq_len=SEQ_LEN, output_len=PRED_LEN, dropout=0.3
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0 
    
    print("\n开始训练...")
    for epoch in range(EPOCHS):
        # 1. Training
        model.train()
        batch_train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
        
        epoch_train_loss = np.mean(batch_train_losses)
        train_losses.append(epoch_train_loss)
        
        # 2. Validation
        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                output = model(batch_x)
                loss = criterion(output, batch_y)
                batch_val_losses.append(loss.item())
        
        epoch_val_loss = np.mean(batch_val_losses)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train: {epoch_train_loss:.6f} | Val: {epoch_val_loss:.6f}", end="")
        
        # 3. Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0 
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(" -> Saved (*)")
        else:
            patience_counter += 1
            print(f" -> Patience {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print(f"\n[早停触发] Val Loss 连续 {PATIENCE} 轮未下降。")
            break
            
    print(f"\n训练结束。最佳验证集 Loss: {best_val_loss:.6f}")
    
    # ================= 最终测试集评估 (Paper Metric) =================
    print("\n--- [Final Evaluation] 测试集最终评估 ---")
    
    # 重新加载最佳模型权重
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    
    test_preds = []
    test_trues = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            output = model(batch_x)
            test_preds.append(output.cpu().numpy())
            test_trues.append(batch_y.numpy())
            
    y_pred = np.concatenate(test_preds, axis=0)
    y_true = np.concatenate(test_trues, axis=0)
    
    # 计算归一化后的指标
    mse_score = mean_squared_error(y_true.flatten(), y_pred.flatten())
    mae_score = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse_score = np.sqrt(mse_score)
    
    print(f"Test Set Results (Scaled):")
    print(f" >> MSE  : {mse_score:.6f}")
    print(f" >> RMSE : {rmse_score:.6f}")
    print(f" >> MAE  : {mae_score:.6f}")

    # ================= 绘图 =================
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='orange', linestyle='--')
    best_epoch_idx = np.argmin(val_losses)
    plt.scatter(best_epoch_idx, best_val_loss, color='red', s=100, zorder=5, label='Best Model')
    plt.title('Training Process')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    print(f"Loss 曲线已保存: {LOSS_PLOT_PATH}")

if __name__ == "__main__":
    train_process()