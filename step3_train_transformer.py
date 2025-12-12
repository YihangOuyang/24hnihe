# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib 
import pvlib 
import warnings
import random
import os

try:
    from step0_model import TransformerNBEATS, PhysicsWeightedMSELoss
except ImportError:
    raise ImportError("❌ 错误: 请确保 step0_model.py 包含 TransformerNBEATS 和 PhysicsWeightedMSELoss 类")

warnings.filterwarnings("ignore")

# ================= 配置区域 =================
CURRENT_DIR = Path(__file__).parent.absolute()
DATA_FILE = CURRENT_DIR / "rawdata" / "Final_Training_Dataset_15min.csv"
PARAM_FILE = CURRENT_DIR / "rawdata" / "physics_params.csv" 

MODEL_SAVE_PATH = CURRENT_DIR / "outputs" / "model"/"transformer_best.pth"
SCALER_X_PATH = CURRENT_DIR / "outputs" / "model"/"scaler_x.pkl"
SCALER_Y_PATH = CURRENT_DIR / "outputs" / "model"/"scaler_y.pkl"
LOSS_PLOT_PATH = CURRENT_DIR / "outputs" / "model"/"training_loss_curve.png"

SEQ_LEN = 96
PRED_LEN = 96
BATCH_SIZE = 32
EPOCHS = 100 
LEARNING_RATE = 1e-4
PATIENCE = 15 
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATITUDE = 37.427963
LONGITUDE = -122.154785
CAPACITY = 30.1 
TARGET_COL = 'power_kw' 

# 仅保留气象特征 (这些可以从天气预报获得)
RAW_FEATURE_COLS = [
    'NWP_GHI', 'NWP_DNI', 'NWP_DHI', 
    'NWP_Temp', 'NWP_Wind', 'NWP_Humidity', 
    'NWP_Cloud', 'NWP_Precip', 'NWP_Press' 
]
# ===============================================================

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class PVDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

def process_time_to_lst(df):
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert('Etc/GMT+8') 
    return df

def align_to_midnight(df):
    midnight = np.where((df.index.hour == 0) & (df.index.minute == 0))[0]
    if len(midnight) == 0: raise ValueError("No midnight found")
    return df.iloc[midnight[0]:]

def feature_engineering(df):
    times = df.index
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz=times.tz)
    solpos = loc.get_solarposition(times)
    df['Solar_El_sin'] = np.sin(np.radians(solpos['elevation'].values))
    df['Solar_Az_sin'] = np.sin(np.radians(solpos['azimuth'].values))
    df['Solar_Az_cos'] = np.cos(np.radians(solpos['azimuth'].values))
    h = times.hour + times.minute / 60.0
    df['Hour_sin'] = np.sin(2 * np.pi * h / 24.0)
    df['Hour_cos'] = np.cos(2 * np.pi * h / 24.0)
    day = times.dayofyear
    df['Day_sin'] = np.sin(2 * np.pi * day / 365)
    df['Day_cos'] = np.cos(2 * np.pi * day / 365)

    return df

def load_physics_params():
    defaults = {'a': 0.2, 'beta': -0.5, 'c': 0.05}
    if not PARAM_FILE.exists(): return defaults['a'], defaults['beta'], defaults['c']
    try:
        df = pd.read_csv(PARAM_FILE)
        p = dict(zip(df['parameter'], df['value']))
        return p['a'], p['beta'], p['c']
    except:
        return defaults['a'], defaults['beta'], defaults['c']

# ===============================================================
# 【修正】纯 NWP 驱动序列生成 (无历史功率依赖)
# ===============================================================
def create_pure_nwp_sequences(data_x_scaled, data_y_scaled, seq_len, pred_len, step):
    """
    构造日前预测序列:
    Input X: Future NWP & Time [t, t+96] 
             (注意：这里虽然叫 'Future'，但在训练时就是对应 target 那一天的天气)
    Target Y: Future Power [t, t+96]
    
    完全不使用 Past Power，符合 Day-Ahead 规则。
    """
    xs, ys = [], []
    total_len = len(data_x_scaled)
    
    # 我们需要取出 [i, i+pred_len] 的天气 作为输入
    # 来预测 [i, i+pred_len] 的功率
    # 所以 i 可以从 0 开始 (只要 NWP 是对齐的)
    
    # 限制范围：保证 i + pred_len 不越界
    for i in range(0, total_len - pred_len + 1, step):
        
        # Input: 未来 24h 的天气预报 + 时间特征
        # 维度: [96, N_Features]
        x_seq = data_x_scaled[i : i + pred_len, :]
        
        # Target: 未来 24h 的真实功率
        y_seq = data_y_scaled[i : i + pred_len, 0] # 0 是因为 scaler_y 只有一列
        
        xs.append(x_seq)
        ys.append(y_seq)
        
    return np.array(xs), np.array(ys)

def check_covariate_shift(train_df, val_df, target_col, feature_cols):
    """
    诊断：检查训练集和验证集是否存在严重的分布偏移 (Covariate Shift)
    重点检查: 
    1. Target Power 分布
    2. 关键气象特征 (GHI, Temp) 分布
    """
    print("\n🔍 [Diagnostics] Checking for Covariate Shift...")
    
    # 创建保存图片的目录
    diag_dir = Path("diagnostics")
    diag_dir.mkdir(exist_ok=True)
    
    # 1. Target Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(train_df[target_col], bins=50, alpha=0.5, density=True, label='Train', color='blue')
    plt.hist(val_df[target_col], bins=50, alpha=0.5, density=True, label='Val', color='orange')
    plt.title(f"Target Distribution: {target_col}")
    plt.xlabel("Power (kW)")
    plt.legend()
    plt.savefig(diag_dir / "shift_target_dist.png")
    print(f"   -> Saved: {diag_dir}/shift_target_dist.png")
    plt.close()
    
    # 2. Key Feature Distributions (GHI & Temp)
    # 检查列表中是否存在这些列
    check_feats = ['NWP_GHI', 'NWP_Temp', 'Solar_El_sin']
    valid_checks = [f for f in check_feats if f in feature_cols]
    
    for feat in valid_checks:
        plt.figure(figsize=(10, 5))
        plt.hist(train_df[feat], bins=50, alpha=0.5, density=True, label='Train', color='blue')
        plt.hist(val_df[feat], bins=50, alpha=0.5, density=True, label='Val', color='orange')
        plt.title(f"Feature Distribution: {feat}")
        plt.legend()
        plt.savefig(diag_dir / f"shift_{feat}.png")
        print(f"   -> Saved: {diag_dir}/shift_{feat}.png")
        plt.close()
        
    print("✅ Diagnosis Complete. Please check the 'diagnostics' folder.\n")

def train_process():
    seed_everything(SEED)
    print(f"=== Training: Pure NWP-Driven Day-Ahead Model ===")
    
    # 1. 数据加载
    if not DATA_FILE.exists(): return print("Data missing")
    df = pd.read_csv(DATA_FILE, index_col=0)
    df.columns = df.columns.str.strip()
    target_col = TARGET_COL if TARGET_COL in df.columns else 'power_kw'
    
    df = process_time_to_lst(df)
    df = align_to_midnight(df)
    df = feature_engineering(df)
    df.dropna(subset=[target_col], inplace=True)

    # 2. 定义特征列 (纯气象 + 时间)
    PHYSICS_COLS = ['Solar_El_sin', 'Solar_Az_sin', 'Solar_Az_cos']
    TIME_COLS = ['Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos']
    
    valid_nwp = [c for c in RAW_FEATURE_COLS if c in df.columns]
    
    # 最终特征 (不包含 Power!)
    FEATURE_COLS = valid_nwp + PHYSICS_COLS + TIME_COLS
    print(f"   Features (Pure Exogenous): {len(FEATURE_COLS)} dims")
    print(f"   Note: Past Power removed to satisfy Day-Ahead constraints.")
    
    # 找到 Solar_El_sin 索引 (用于 Loss)
    SOLAR_EL_IDX = FEATURE_COLS.index('Solar_El_sin')
    
    # 3. 数据划分 & 归一化
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end] # 这里需要显式定义 val_df 用于画图
    # test_df = df.iloc[val_end:]
    
    # === [新增] 插入诊断代码 ===
    check_covariate_shift(train_df, val_df, target_col, FEATURE_COLS)

    scaler_x = StandardScaler()
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    # Fit
    scaler_x.fit(train_df[FEATURE_COLS].values)
    scaler_y.fit(train_df[[target_col]].values)
    
    joblib.dump(scaler_x, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    
    # Transform Full Data
    data_x_scaled = scaler_x.transform(df[FEATURE_COLS].values)
    data_y_scaled = scaler_y.transform(df[[target_col]].values)
    
    # 4. 序列生成
    print("   [Sequence Construction] Generating Pure NWP sequences...")
    
    # 切片
    # 注意: create_pure_nwp_sequences 不需要历史数据做 Lag，所以不需要像之前那样留 buffer
    # 但为了逻辑一致，我们还是按索引切
    x_train_raw = data_x_scaled[:train_end]
    y_train_raw = data_y_scaled[:train_end]
    
    x_val_raw   = data_x_scaled[train_end:val_end]
    y_val_raw   = data_y_scaled[train_end:val_end]
    
    x_test_raw  = data_x_scaled[val_end:]
    y_test_raw  = data_y_scaled[val_end:]
    
    x_train, y_train = create_pure_nwp_sequences(x_train_raw, y_train_raw, SEQ_LEN, PRED_LEN, step=1)
    x_val, y_val     = create_pure_nwp_sequences(x_val_raw,   y_val_raw,   SEQ_LEN, PRED_LEN, step=96)
    x_test, y_test   = create_pure_nwp_sequences(x_test_raw,  y_test_raw,  SEQ_LEN, PRED_LEN, step=96)
    
    print(f"   Train Samples: {x_train.shape}")
    
    # 5. DataLoader
    train_loader = DataLoader(PVDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(PVDataset(x_val, y_val),   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(PVDataset(x_test, y_test),  batch_size=BATCH_SIZE, shuffle=False)

    # 6. 模型初始化
    # Input Dim = Features 数量
    model = TransformerNBEATS(
        num_stacks=1, num_blocks_per_stack=1, input_dim=len(FEATURE_COLS), 
        d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128, 
        input_seq_len=SEQ_LEN, output_len=PRED_LEN, dropout=0.4 
    ).to(DEVICE)
    
    a, beta, c = load_physics_params()
    criterion = PhysicsWeightedMSELoss(a, beta, c, capacity=CAPACITY).to(DEVICE)
    # criterion = nn.MSELoss().to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    # Cosine Annealing (帮助跳出局部最优)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # 7. 训练循环
    best_loss = float('inf')
    patience_cnt = 0
    train_loss_hist, val_loss_hist = [], []
    
    print("\n>>> Start Training <<<")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            if out.dim() == 3: out = out.squeeze(-1)
            
            # Physics Proxy (Solar_El)
            physics_proxy = bx[:, :, SOLAR_EL_IDX]
            
            loss = criterion(out, by, physics_proxy)
            # loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                if out.dim() == 3: out = out.squeeze(-1)
                physics_proxy = bx[:, :, SOLAR_EL_IDX]
                val_loss += criterion(out, by, physics_proxy).item()
                # val_loss += criterion(out, by).item()
        val_loss /= len(val_loader)
        
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        
        print(f"Epoch {epoch+1:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e}", end="")
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(" -> Saved (*)")
        else:
            patience_cnt += 1
            print(f" -> Patience {patience_cnt}/{PATIENCE}")
            if patience_cnt >= PATIENCE: break

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_hist, label='Train')
    plt.plot(val_loss_hist, label='Val')
    plt.title('Loss Curve (Pure NWP)')
    plt.legend()
    plt.savefig(LOSS_PLOT_PATH)

    # Final Eval
    print("\n--- Final Test ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            out = model(bx)
            if out.dim() == 3: out = out.squeeze(-1)
            preds.append(out.cpu().numpy())
            trues.append(by.numpy())
            
    p_flat = scaler_y.inverse_transform(np.concatenate(preds).reshape(-1,1)).flatten()
    t_flat = scaler_y.inverse_transform(np.concatenate(trues).reshape(-1,1)).flatten()
    p_flat = np.maximum(p_flat, 0)
    
    rmse = np.sqrt(mean_squared_error(t_flat, p_flat))
    nrmse = rmse/CAPACITY * 100
    mae = mean_absolute_error(t_flat, p_flat)
    print(f"RMSE: {rmse:.4f} kW | nRMSE: {nrmse:.2f}% | MAE: {mae:.4f} kW")

if __name__ == "__main__":
    train_process()