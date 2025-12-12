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

# 引入 step0 中的 Transformer 模型 (Loss 我们在本地重写以适配新公式)
try:
    from step0_model import TransformerNBEATS
except ImportError:
    raise ImportError("❌ 错误: 请确保 step0_model.py 包含 TransformerNBEATS 类")

warnings.filterwarnings("ignore")

# ================= 配置区域 =================
CURRENT_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR = CURRENT_DIR / "outputs" / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = CURRENT_DIR / "rawdata" / "Final_Training_Dataset_15min.csv"

MODEL_SAVE_PATH = OUTPUT_DIR / "transformer_best.pth"
SCALER_X_PATH = OUTPUT_DIR / "scaler_x.pkl"
SCALER_Y_PATH = OUTPUT_DIR / "scaler_y.pkl"
LOSS_PLOT_PATH = OUTPUT_DIR / "training_loss_curve.png"

SEQ_LEN = 96
PRED_LEN = 96
BATCH_SIZE = 64
EPOCHS = 100 
LEARNING_RATE = 5e-4
PATIENCE = 20 
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATITUDE = 37.427963
LONGITUDE = -122.154785
CAPACITY = 30.1 

PHYSICS_TIME_SHIFT_MINUTES = 0
RAW_TARGET_COL = 'power_kw' 
TRAIN_TARGET_COL = 'k_index'

RAW_FEATURE_COLS = [
    'NWP_GHI', 'NWP_DNI', 'NWP_DHI', 
    'NWP_Temp', 'NWP_Wind', 'NWP_Humidity', 
    'NWP_Cloud', 'NWP_Precip', 'NWP_Press' 
]

# [NEW] 你的物理拟合参数
PHYSICS_PARAMS = {
    'a': 0.040,
    'beta': -0.656,
    'gamma': 1.154
}
# ===============================================================

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# [修改 1] Dataset 增加 p_clear 返回
class PVDataset(Dataset):
    def __init__(self, x, y, p_clear):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
        self.p_clear = torch.FloatTensor(p_clear) # 新增：晴空功率用于计算物理权重
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx], self.p_clear[idx]

# [修改 2] 定义物理加权 Loss (Cut-off Power Law)
class PhysicsWeightedMSELoss(nn.Module):
    def __init__(self, a, beta, gamma, capacity, device):
        super().__init__()
        self.a = a
        self.beta = beta
        self.gamma = gamma
        self.capacity = capacity
        self.device = device
        
    def forward(self, pred_k, target_k, p_clearsky):
        """
        pred_k: 模型预测的 k_index (Normalized)
        target_k: 真实的 k_index (Normalized)
        p_clearsky: 对应的晴空功率 (kW)
        """
        # 1. 还原物理功率 P_real (kW)
        #    使用 Target 来计算权重通常更稳定 (Aleatoric Uncertainty of the true state)
        p_est_kw = target_k * p_clearsky 
        
        # 2. 计算 P (p.u.) 归一化功率 [0, 1]
        p_pu = p_est_kw / self.capacity
        # 截断以防止 log(0) 或负数导致 NaN，同时保留 (1-P) 的有效性
        p_pu = torch.clamp(p_pu, 0.001, 0.999) 
        
        # 3. 代入你的物理公式计算 I_F (波动强度)
        #    Formula: I_F = a * P^beta * (1-P)^gamma
        i_f = self.a * (p_pu ** self.beta) * ((1 - p_pu) ** self.gamma)
        
        # 4. 计算权重 (Weight = 1 / Variance = 1 / I_F^2)
        #    I_F 越大(波动越大)，权重越小；I_F 越小(越稳定)，权重越大
        epsilon = 1e-6
        raw_weight = 1.0 / (i_f**2 + epsilon)
        raw_weight = torch.clamp(raw_weight, max=100.0)
        # 5. 归一化权重 (保持 Loss 数值尺度稳定)
        weight = raw_weight / raw_weight.mean()
        
        # 6. 计算加权 MSE
        #    注意：我们是在 k_index 空间计算 Loss，但用物理空间计算权重
        mse_loss = (pred_k - target_k) ** 2
        weighted_loss = weight * mse_loss
        
        final_loss = 0.5 * weighted_loss + 0.5 * mse_loss
        
        return final_loss.mean()

def process_time_to_lst(df):
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert('Etc/GMT+8') 
    return df

def align_to_midnight(df):
    midnight = np.where((df.index.hour == 0) & (df.index.minute == 0))[0]
    if len(midnight) == 0: raise ValueError("No midnight found")
    return df.iloc[midnight[0]:]

def feature_engineering(df):
    print(f"   [Feature Engineering] Calculating k* with time shift: {PHYSICS_TIME_SHIFT_MINUTES} min...")
    times = df.index
    times_phys = times + pd.Timedelta(minutes=PHYSICS_TIME_SHIFT_MINUTES)
    
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz=times.tz)
    solpos = loc.get_solarposition(times_phys)
    
    df['Solar_El_sin'] = np.sin(np.radians(solpos['elevation'].values))
    df['Solar_Az_sin'] = np.sin(np.radians(solpos['azimuth'].values))
    df['Solar_Az_cos'] = np.cos(np.radians(solpos['azimuth'].values))
    
    h = times.hour + times.minute / 60.0
    df['Hour_sin'] = np.sin(2 * np.pi * h / 24.0)
    df['Hour_cos'] = np.cos(2 * np.pi * h / 24.0)
    day = times.dayofyear
    df['Day_sin'] = np.sin(2 * np.pi * day / 365.25)
    df['Day_cos'] = np.cos(2 * np.pi * day / 365.25)
    
    cs = loc.get_clearsky(times_phys)
    estimated_efficiency = 1.0 
    df['P_clearsky'] = cs['ghi'] * (CAPACITY / 1000.0) * estimated_efficiency
    
    epsilon = 0.1 
    df['k_index'] = df[RAW_TARGET_COL] / (df['P_clearsky'] + epsilon)
    
    is_night = solpos['elevation'] < 5 
    df.loc[is_night, 'k_index'] = 0.0
    df.loc[is_night, 'P_clearsky'] = 0.0
    df['k_index'] = df['k_index'].clip(0.0, 1.5)
    
    return df

# [修改 3] 序列生成函数现在需要处理 P_clearsky
def create_sequences_with_physics(data_x_scaled, data_y_scaled, data_p_clear, seq_len, pred_len, step):
    xs, ys, ps = [], [], []
    total_len = len(data_x_scaled)
    for i in range(0, total_len - pred_len + 1, step):
        x_seq = data_x_scaled[i : i + pred_len, :]
        y_seq = data_y_scaled[i : i + pred_len, 0]
        # P_clearsky 不需要归一化，直接取物理值即可
        p_seq = data_p_clear[i : i + pred_len]
        
        xs.append(x_seq)
        ys.append(y_seq)
        ps.append(p_seq)
    return np.array(xs), np.array(ys), np.array(ps)

def check_covariate_shift(train_df, val_df, target_col, feature_cols):
    print("\n🔍 [Diagnostics] Checking for Covariate Shift...")
    diag_dir = OUTPUT_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.hist(train_df[target_col], bins=50, alpha=0.5, density=True, label='Train', color='blue')
    plt.hist(val_df[target_col], bins=50, alpha=0.5, density=True, label='Val', color='orange')
    plt.title(f"Target Distribution: {target_col}")
    plt.legend()
    plt.savefig(diag_dir / "shift_target_dist.png")
    plt.close()
    print("✅ Diagnosis Complete.\n")

def train_process():
    seed_everything(SEED)
    print(f"=== Training: Physics-Guided Transformer (Cut-off Power Law) ===")
    
    if not DATA_FILE.exists(): return print("Data missing")
    df = pd.read_csv(DATA_FILE, index_col=0)
    df.columns = df.columns.str.strip()
    
    df = process_time_to_lst(df)
    df = align_to_midnight(df)
    df = feature_engineering(df)
    df.dropna(subset=[RAW_TARGET_COL], inplace=True)

    PHYSICS_COLS = ['Solar_El_sin', 'Solar_Az_sin', 'Solar_Az_cos']
    TIME_COLS = ['Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos']
    valid_nwp = [c for c in RAW_FEATURE_COLS if c in df.columns]
    FEATURE_COLS = valid_nwp + PHYSICS_COLS + TIME_COLS
    
    # Interleaved Split
    start_date = df.index[0]
    days_diff = (df.index - start_date).days
    biweek_idx = days_diff // 14
    cycle_idx = biweek_idx % 6
    
    train_mask = np.isin(cycle_idx, [0, 1, 2, 3])
    val_mask   = (cycle_idx == 4)
    test_mask  = (cycle_idx == 5)
    
    train_df = df[train_mask].copy()
    val_df   = df[val_mask].copy()
    test_df  = df[test_mask].copy()
    
    check_covariate_shift(train_df, val_df, TRAIN_TARGET_COL, FEATURE_COLS)
    
    scaler_x = StandardScaler()
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    scaler_x.fit(train_df[FEATURE_COLS].values)
    scaler_y.fit(train_df[[TRAIN_TARGET_COL]].values)
    
    joblib.dump(scaler_x, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    
    def transform_df(d):
        x = scaler_x.transform(d[FEATURE_COLS].values)
        y = scaler_y.transform(d[[TRAIN_TARGET_COL]].values)
        return x, y

    x_train_sc, y_train_sc = transform_df(train_df)
    x_val_sc,   y_val_sc   = transform_df(val_df)
    x_test_sc,  y_test_sc  = transform_df(test_df)
    
    # [修改 4] 提取 P_clearsky 列
    p_train_raw = train_df['P_clearsky'].values
    p_val_raw   = val_df['P_clearsky'].values
    p_test_raw  = test_df['P_clearsky'].values
    
    print("   [Sequence Construction] Generating sequences with Physics Info...")
    # 传入 P_clearsky 生成序列
    x_train, y_train, p_train = create_sequences_with_physics(x_train_sc, y_train_sc, p_train_raw, SEQ_LEN, PRED_LEN, step=1)
    x_val,   y_val,   p_val   = create_sequences_with_physics(x_val_sc,   y_val_sc,   p_val_raw,   SEQ_LEN, PRED_LEN, step=96)
    x_test,  y_test,  p_test  = create_sequences_with_physics(x_test_sc,  y_test_sc,  p_test_raw,  SEQ_LEN, PRED_LEN, step=96)
    
    # DataLoader 传入 3 个 Tensor
    train_loader = DataLoader(PVDataset(x_train, y_train, p_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(PVDataset(x_val,   y_val,   p_val),   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(PVDataset(x_test,  y_test,  p_test),  batch_size=BATCH_SIZE, shuffle=False)

    model = TransformerNBEATS(
        num_stacks=1, num_blocks_per_stack=1, input_dim=len(FEATURE_COLS), 
        d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32, 
        input_seq_len=SEQ_LEN, output_len=PRED_LEN, dropout=0.5 
    ).to(DEVICE)
    
    # [修改 5] 使用 PhysicsWeightedMSELoss
    # 使用你拟合出的参数: a=0.040, beta=-0.656, gamma=1.154
    print(f"   [Loss Function] Initializing Physics-Guided Loss with a={PHYSICS_PARAMS['a']}, beta={PHYSICS_PARAMS['beta']}, gamma={PHYSICS_PARAMS['gamma']}")
    
    criterion = PhysicsWeightedMSELoss(
        a=PHYSICS_PARAMS['a'], 
        beta=PHYSICS_PARAMS['beta'], 
        gamma=PHYSICS_PARAMS['gamma'], 
        capacity=CAPACITY, 
        device=DEVICE
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_loss = float('inf')
    patience_cnt = 0
    train_loss_hist, val_loss_hist = [], []
    
    print("\n>>> Start Training (Physics-Guided) <<<")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        # 解包 bx, by, bp
        for bx, by, bp in train_loader:
            bx, by, bp = bx.to(DEVICE), by.to(DEVICE), bp.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            if out.dim() == 3: out = out.squeeze(-1)
            
            # 传入 P_clearsky 计算物理 Loss
            loss = criterion(out, by, bp)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by, bp in val_loader:
                bx, by, bp = bx.to(DEVICE), by.to(DEVICE), bp.to(DEVICE)
                out = model(bx)
                if out.dim() == 3: out = out.squeeze(-1)
                # Validation 也使用物理权重，评价更公正
                val_loss += criterion(out, by, bp).item()
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

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_hist, label='Train')
    plt.plot(val_loss_hist, label='Val')
    plt.title('Loss Curve (Physics-Weighted)')
    plt.legend()
    plt.savefig(LOSS_PLOT_PATH)

    print("\n--- Final Test (Restoring Power) ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    preds, trues = [], []
    
    # 提取测试集的 P_clearsky (用于还原功率计算指标)
    # p_test 已经在上面生成了，且和 x_test 对齐
    p_clear_test_list = [] 
    
    with torch.no_grad():
        for bx, by, bp in test_loader:
            bx = bx.to(DEVICE)
            out = model(bx)
            if out.dim() == 3: out = out.squeeze(-1)
            preds.append(out.cpu().numpy())
            trues.append(by.numpy())
            p_clear_test_list.append(bp.numpy())
    
    # 拼接
    preds_k_scaled = np.concatenate(preds)
    trues_k_scaled = np.concatenate(trues)
    p_clear_test_all = np.concatenate(p_clear_test_list)
    
    # 1. 反归一化 k_index
    preds_k_real = scaler_y.inverse_transform(preds_k_scaled.reshape(-1, 1)).reshape(preds_k_scaled.shape)
    trues_k_real = scaler_y.inverse_transform(trues_k_scaled.reshape(-1, 1)).reshape(trues_k_scaled.shape)
    
    # 2. 还原功率 P = k * P_clear
    #    注意：这里的 p_clear_test_all 已经是序列化的，直接相乘即可
    p_pred = preds_k_real * p_clear_test_all
    p_true = trues_k_real * p_clear_test_all
    
    # 3. 物理约束
    p_pred = np.maximum(p_pred, 0.0)
    p_true = np.maximum(p_true, 0.0)
    
    # 4. 计算指标
    rmse = np.sqrt(mean_squared_error(p_true.flatten(), p_pred.flatten()))
    nrmse = rmse/CAPACITY * 100
    mae = mean_absolute_error(p_true.flatten(), p_pred.flatten())
    
    print(f"RMSE: {rmse:.4f} kW")
    print(f"nRMSE: {nrmse:.2f} %")
    print(f"MAE: {mae:.4f} kW")

if __name__ == "__main__":
    train_process()