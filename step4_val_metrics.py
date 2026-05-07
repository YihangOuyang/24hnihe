# -*- coding: utf-8 -*-
"""
Step 4: Benchmark Analysis (Final Version - with Standard Transformer and PiT-Net)
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import joblib
from pathlib import Path
import pvlib
import warnings
import sys
import os

# --- Import Our Model ---
try:
    from step0_model import TransformerNBEATS
except ImportError:
    print("❌ Error: 'step0_model.py' not found.")
    sys.exit()

warnings.filterwarnings("ignore")

# ================= Configuration =================
CURRENT_DIR = Path(__file__).parent.absolute()
DATA_FILE = CURRENT_DIR / "rawdata" / "Final_Training_Dataset_15min.csv"
OUTPUT_DIR = CURRENT_DIR / "outputs" / "benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = CURRENT_DIR / "outputs" / "model"

# 【修改点 1】: 同时指定纯数据模型和你的物理模型的权重路径
TRANSFORMER_STD_PATH = MODEL_DIR / "transformer_best1_0.pth"        # Standard Transformer
TRANSFORMER_PROPOSED_PATH = MODEL_DIR / "transformer_best0.5_0.5.pth" # PiT-Net (Proposed)

SCALER_X_PATH = MODEL_DIR / "scaler_x.pkl"
SCALER_Y_PATH = MODEL_DIR / "scaler_y.pkl"

SEQ_LEN = 96
PRED_LEN = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAPACITY = 30.1
LATITUDE = 37.427963
LONGITUDE = -122.154785
TIMEZONE_FIXED = "Etc/GMT+8"
PHYSICS_TIME_SHIFT_MINUTES = 0
RAW_TARGET_COL = 'power_kw'

RAW_FEATURE_COLS = [
    'NWP_GHI', 'NWP_DNI', 'NWP_DHI', 
    'NWP_Temp', 'NWP_Wind', 'NWP_Humidity', 
    'NWP_Cloud', 'NWP_Precip', 'NWP_Press' 
]

# Switches
FORCE_RETRAIN_BENCHMARK = False  # False=Smart Check, True=Force Retrain
PLOT_DAY_INDEX = 20

# ================= Data Pipeline =================

def process_time_to_lst(df):
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert(TIMEZONE_FIXED)
    return df

def align_to_midnight(df):
    midnight = np.where((df.index.hour == 0) & (df.index.minute == 0))[0]
    if len(midnight) == 0: raise ValueError("No midnight found")
    return df.iloc[midnight[0]:]

def feature_engineering(df):
    times = df.index
    times_phys = times + pd.Timedelta(minutes=PHYSICS_TIME_SHIFT_MINUTES)
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz=times.tz)
    solpos = loc.get_solarposition(times_phys)
    
    # Physics
    df['Solar_El_sin'] = np.sin(np.radians(solpos['elevation'].values))
    df['Solar_Az_sin'] = np.sin(np.radians(solpos['azimuth'].values))
    df['Solar_Az_cos'] = np.cos(np.radians(solpos['azimuth'].values))
    
    # Time
    h = times.hour + times.minute / 60.0
    df['Hour_sin'] = np.sin(2 * np.pi * h / 24.0)
    df['Hour_cos'] = np.cos(2 * np.pi * h / 24.0)
    day = times.dayofyear
    df['Day_sin'] = np.sin(2 * np.pi * day / 365.25)
    df['Day_cos'] = np.cos(2 * np.pi * day / 365.25)
    
    # P_clearsky
    cs = loc.get_clearsky(times_phys)
    df['P_clearsky'] = cs['ghi'] * (CAPACITY / 1000.0)
    
    # Night Mask
    is_night = solpos['elevation'] < 5 
    df.loc[is_night, 'P_clearsky'] = 0.0 
    
    return df

def load_data_strict():
    print("--- Loading Data (Strict Pipeline) ---")
    df = pd.read_csv(DATA_FILE, index_col=0)
    df.columns = df.columns.str.strip()
    
    df = process_time_to_lst(df)
    df = align_to_midnight(df)
    df = feature_engineering(df)
    df.dropna(subset=[RAW_TARGET_COL], inplace=True)
    
    valid_nwp = [c for c in RAW_FEATURE_COLS if c in df.columns]
    FEATURE_COLS = valid_nwp + ['Solar_El_sin', 'Solar_Az_sin', 'Solar_Az_cos', 
                                'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos']
    
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y_k = joblib.load(SCALER_Y_PATH)
    
    X_scaled = scaler_x.transform(df[FEATURE_COLS].values)
    Y_raw_power = df[RAW_TARGET_COL].values.reshape(-1, 1)
    P_clear_arr = df['P_clearsky'].values.reshape(-1, 1)
    
    start_date = df.index[0]
    days_diff = (df.index - start_date).days
    biweek_idx = days_diff // 14
    cycle_idx = biweek_idx % 6
    
    train_mask = np.isin(cycle_idx, [0, 1, 2, 3, 4]) 
    test_mask = (cycle_idx == 5)
    
    print(f"   Train Samples: {np.sum(train_mask)}")
    print(f"   Test Samples:  {np.sum(test_mask)}")
    
    return X_scaled, Y_raw_power, P_clear_arr, train_mask, test_mask, scaler_y_k, len(FEATURE_COLS), df.index

def create_sequences(data_x, data_y, data_p, seq_len, pred_len, step):
    xs, ys, ps = [], [], []
    for i in range(0, len(data_x) - pred_len + 1, step):
        xs.append(data_x[i : i+pred_len, :])
        ys.append(data_y[i : i+pred_len, 0])
        ps.append(data_p[i : i+pred_len, 0])
    return np.array(xs), np.array(ys), np.array(ps)

# ================= Models =================

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_len, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class MLPModel(nn.Module):
    def __init__(self, input_len, input_dim, output_len):
        super(MLPModel, self).__init__()
        self.flatten_dim = input_len * input_dim
        self.net = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(self.flatten_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), 
            nn.Linear(128, output_len)
        )
    def forward(self, x):
        return self.net(x)

class CNNModel(nn.Module):
    def __init__(self, input_len, input_dim, output_len):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Linear(128, output_len)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# ================= Smart Logic Functions =================

def train_or_load_sklearn(name, model_class, X_tr_flat, Y_tr_flat, path, **kwargs):
    if not FORCE_RETRAIN_BENCHMARK and path.exists():
        print(f"   🔹 [Skipped] Loading existing {name}...")
        model = joblib.load(path)
    else:
        print(f"   🔸 [Training] Fitting {name} (Force={FORCE_RETRAIN_BENCHMARK})...")
        model = model_class(**kwargs)
        model.fit(X_tr_flat, Y_tr_flat)
        joblib.dump(model, path)
        print(f"      Saved to {path.name}")
    return model

def train_or_load_torch(name, model_obj, X_tr, Y_tr, path, epochs=5, batch_size=128):
    path = Path(path)
    load_success = False
    
    if not FORCE_RETRAIN_BENCHMARK and path.exists():
        try:
            print(f"   🔹 [Skipped] Loading existing {name}...")
            model_obj.load_state_dict(torch.load(path, map_location=DEVICE))
            load_success = True
        except RuntimeError as e:
            print(f"   ⚠️ [Load Failed] Architecture mismatch for {name}. Retraining automatically.")
    
    if not load_success:
        print(f"   🔸 [Training] Fitting {name} (Epochs={epochs})...")
        crit = nn.MSELoss()
        opt = torch.optim.Adam(model_obj.parameters(), lr=1e-3)
        dset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(Y_tr))
        loader = DataLoader(dset, batch_size=batch_size, shuffle=True)
        
        model_obj.train()
        for ep in range(epochs):
            total_loss = 0
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                opt.zero_grad()
                loss = crit(model_obj(bx), by)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if (ep+1) % 2 == 0:
                print(f"      Ep {ep+1}/{epochs} Loss: {total_loss/len(loader):.4f}")
        
        torch.save(model_obj.state_dict(), path)
        print(f"      Saved to {path.name}")
    
    return model_obj

# ================= Main Execution =================

def run_benchmark():
    X_all, Y_all, P_all, train_mask, test_mask, scaler_y_k, n_feat, all_times = load_data_strict()
    
    print("\n--- Creating Sequences ---")
    X_tr, Y_tr, _ = create_sequences(X_all[train_mask], Y_all[train_mask], P_all[train_mask], SEQ_LEN, PRED_LEN, 1)
    X_te, Y_te, P_te = create_sequences(X_all[test_mask], Y_all[test_mask], P_all[test_mask], SEQ_LEN, PRED_LEN, 96)
    
    test_indices = np.arange(len(X_all))[test_mask]
    test_timestamps = []
    valid_indices = test_indices[: len(X_all[test_mask]) - PRED_LEN + 1 : 96]
    for idx in valid_indices: test_timestamps.append(all_times[idx : idx+PRED_LEN])
    
    limit = min(len(X_te), len(test_timestamps))
    X_te, Y_te, P_te, test_timestamps = X_te[:limit], Y_te[:limit], P_te[:limit], test_timestamps[:limit]
    
    print(f"   Test Batches: {len(X_te)}")
    
    scaler_power = MinMaxScaler()
    Y_tr_scaled = scaler_power.fit_transform(Y_tr.reshape(-1, 1)).reshape(Y_tr.shape)
    
    X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)
    X_te_flat = X_te.reshape(X_te.shape[0], -1)
    Y_tr_flat = Y_tr_scaled 
    
    results = {}
    print("\n>>> 🚀 Start Benchmark Training/Loading <<<")

    # --- 1-6. 各种基线模型 ---
    model_ridge = train_or_load_sklearn("Ridge", Ridge, X_tr_flat, Y_tr_flat, OUTPUT_DIR / "ridge.pkl", alpha=1.0)
    results['Ridge'] = scaler_power.inverse_transform(model_ridge.predict(X_te_flat).reshape(-1, 1)).reshape(Y_te.shape)

    model_rf = train_or_load_sklearn("RandomForest", RandomForestRegressor, X_tr_flat, Y_tr_flat, OUTPUT_DIR / "rf.pkl", n_estimators=20, n_jobs=-1, max_depth=10)
    results['RandomForest'] = scaler_power.inverse_transform(model_rf.predict(X_te_flat).reshape(-1, 1)).reshape(Y_te.shape)

    model_xgb = train_or_load_sklearn("XGBoost", xgb.XGBRegressor, X_tr_flat, Y_tr_flat, OUTPUT_DIR / "xgb.pkl", n_estimators=100, max_depth=6, device="cuda" if torch.cuda.is_available() else "cpu")
    results['XGBoost'] = scaler_power.inverse_transform(model_xgb.predict(X_te_flat).reshape(-1, 1)).reshape(Y_te.shape)

    model_mlp = MLPModel(SEQ_LEN, n_feat, PRED_LEN).to(DEVICE)
    model_mlp = train_or_load_torch("MLP", model_mlp, X_tr, Y_tr_scaled, OUTPUT_DIR / "mlp.pth", epochs=30)
    model_mlp.eval()
    with torch.no_grad():
        p = []
        for i in range(0, len(X_te), 64):
            batch = torch.FloatTensor(X_te[i:i+64]).to(DEVICE)
            p.append(model_mlp(batch).cpu().numpy())
    results['MLP'] = scaler_power.inverse_transform(np.concatenate(p).reshape(-1, 1)).reshape(Y_te.shape)

    model_lstm = LSTMModel(n_feat, 64, PRED_LEN).to(DEVICE)
    model_lstm = train_or_load_torch("LSTM", model_lstm, X_tr, Y_tr_scaled, OUTPUT_DIR / "lstm.pth", epochs=30)
    model_lstm.eval()
    with torch.no_grad():
        p = []
        for i in range(0, len(X_te), 64):
            batch = torch.FloatTensor(X_te[i:i+64]).to(DEVICE)
            p.append(model_lstm(batch).cpu().numpy())
    results['LSTM'] = scaler_power.inverse_transform(np.concatenate(p).reshape(-1, 1)).reshape(Y_te.shape)

    model_cnn = CNNModel(SEQ_LEN, n_feat, PRED_LEN).to(DEVICE)
    model_cnn = train_or_load_torch("CNN", model_cnn, X_tr, Y_tr_scaled, OUTPUT_DIR / "cnn.pth", epochs=30)
    model_cnn.eval()
    with torch.no_grad():
        p = []
        for i in range(0, len(X_te), 64):
            batch = torch.FloatTensor(X_te[i:i+64]).to(DEVICE)
            p.append(model_cnn(batch).cpu().numpy())
    results['CNN'] = scaler_power.inverse_transform(np.concatenate(p).reshape(-1, 1)).reshape(Y_te.shape)

    # 【修改点 2】: 单独加载 Standard Transformer (无物理约束版)
    print("   🔹 Loading Standard Transformer (Baseline)...")
    try:
        model_tf_std = TransformerNBEATS(
            num_stacks=1, num_blocks_per_stack=1, input_dim=n_feat, 
            d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32, 
            input_seq_len=SEQ_LEN, output_len=PRED_LEN, dropout=0.5
        ).to(DEVICE)
        model_tf_std.load_state_dict(torch.load(TRANSFORMER_STD_PATH, map_location=DEVICE))
        model_tf_std.eval()
        
        k_preds = []
        with torch.no_grad():
            for i in range(0, len(X_te), 64):
                batch = torch.FloatTensor(X_te[i:i+64]).to(DEVICE)
                k_preds.append(model_tf_std(batch).cpu().numpy())
        
        k_preds_scaled = np.concatenate(k_preds)
        k_preds_real = scaler_y_k.inverse_transform(k_preds_scaled.reshape(-1, 1)).reshape(Y_te.shape)
        results['Transformer'] = k_preds_real * P_te 
        
    except Exception as e:
        print(f"❌ Standard Transformer Load Failed: {e}")

    # 【修改点 3】: 加载 PiT-Net (提出的物理加权版)
    print("   🔹 Loading PiT-Net (Proposed)...")
    try:
        model_tf_prop = TransformerNBEATS(
            num_stacks=1, num_blocks_per_stack=1, input_dim=n_feat, 
            d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32, 
            input_seq_len=SEQ_LEN, output_len=PRED_LEN, dropout=0.5
        ).to(DEVICE)
        model_tf_prop.load_state_dict(torch.load(TRANSFORMER_PROPOSED_PATH, map_location=DEVICE))
        model_tf_prop.eval()
        
        k_preds = []
        with torch.no_grad():
            for i in range(0, len(X_te), 64):
                batch = torch.FloatTensor(X_te[i:i+64]).to(DEVICE)
                k_preds.append(model_tf_prop(batch).cpu().numpy())
        
        k_preds_scaled = np.concatenate(k_preds)
        k_preds_real = scaler_y_k.inverse_transform(k_preds_scaled.reshape(-1, 1)).reshape(Y_te.shape)
        results['PiT-Net'] = k_preds_real * P_te 
        
    except Exception as e:
        print(f"❌ PiT-Net Load Failed: {e}")


    # ==========================================
    # Part C: Metrics & Visualization
    # ==========================================
    print("\n📊 Calculating Metrics...")
    metrics_list = []
    step_rmse_dict = {}
    
    Y_real = Y_te.copy() 
    mask_physics_night = (P_te == 0)
    
    n_fixed = np.sum((Y_real[mask_physics_night] > 0.1))
    print(f"🔧 [Physics Fix] Forcing {n_fixed} points in Ground Truth to 0.")
    
    Y_real[mask_physics_night] = 0.0
    
    for name, pred in results.items():
        pred = np.maximum(pred, 0.0)
        pred[mask_physics_night] = 0.0
        
        rmse = np.sqrt(mean_squared_error(Y_real.flatten(), pred.flatten()))
        mae = mean_absolute_error(Y_real.flatten(), pred.flatten())
        r2 = r2_score(Y_real.flatten(), pred.flatten())
        
        metrics_list.append({
            'Model': name, 'RMSE': rmse, 'nRMSE (%)': (rmse/CAPACITY)*100, 'MAE': mae, 'R2': r2
        })
        results[name] = pred
        
        step_mse = np.mean((Y_real - pred) ** 2, axis=0)
        step_rmse = np.sqrt(step_mse)
        step_rmse_dict[name] = step_rmse

    df_metrics = pd.DataFrame(metrics_list).sort_values(by='RMSE')
    print("\n", df_metrics)
    df_metrics.to_csv(OUTPUT_DIR / "benchmark_metrics_final.csv", index=False)
    
    print("\n💾 Saving detailed time-series data...")

    ts_flat = []
    for batch_ts in test_timestamps:
        ts_flat.extend(batch_ts)
    
    df_all_data = pd.DataFrame({
        'Timestamp': ts_flat,
        'Actual_Power_Fixed': Y_real.flatten()
    })

    # 将 8 个模型（包括 Transformer 和 PiT-Net）的预测全部加入 CSV
    for name, pred_matrix in results.items():
        df_all_data[f'Pred_{name}'] = pred_matrix.flatten()

    save_path_ts = OUTPUT_DIR / "benchmark_all_predictions_timeseries.csv"
    df_all_data.to_csv(save_path_ts, index=False)
    print(f"✅ [1/2] Time-series data saved to: {save_path_ts}")

    df_step_rmse = pd.DataFrame(step_rmse_dict)
    df_step_rmse.insert(0, 'Horizon_Hours', np.arange(1, 97) * 0.25)
    
    save_path_step = OUTPUT_DIR / "benchmark_stepwise_rmse_data.csv"
    df_step_rmse.to_csv(save_path_step, index=False)
    print(f"✅ [2/2] Step-wise RMSE data saved to: {save_path_step}")
    
    # 【修改点 4】: 图例颜色分配，PiT-Net 使用抢眼的红色，Standard Transformer 使用粉/紫色
    colors = {
        'PiT-Net': '#d62728',         # 红色，突出显示
        'Transformer': '#e377c2',     # 粉色，作为无物理约束基线
        'XGBoost': '#2ca02c', 'RandomForest': '#8c564b',
        'LSTM': '#1f77b4', 'CNN': '#9467bd', 'MLP': '#ff7f0e', 'Ridge': '#7f7f7f'
    }

    plt.figure(figsize=(10, 6))
    time_steps = np.arange(1, 97) * 15 / 60 
    
    for name, rmse_curve in step_rmse_dict.items():
        c = colors.get(name, 'gray')
        lw = 3 if name == 'PiT-Net' else 1.5
        ls = '-' if name == 'PiT-Net' else '-'
        alpha = 1.0 if name == 'PiT-Net' else 0.7
        plt.plot(time_steps, rmse_curve, color=c, lw=lw, ls=ls, label=name, alpha=alpha)
        
    plt.title("Forecast Horizon Accuracy (Step-wise RMSE)", fontsize=14)
    plt.ylabel("RMSE (kW)", fontsize=12)
    plt.xlabel("Forecast Horizon (Hours)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 24)
    
    save_path_rmse = OUTPUT_DIR / "benchmark_stepwise_rmse.png"
    plt.savefig(save_path_rmse, dpi=300, bbox_inches='tight')

    if PLOT_DAY_INDEX < len(results['PiT-Net']):
        idx = PLOT_DAY_INDEX
        ts = test_timestamps[idx].tz_localize(None)
        
        plt.figure(figsize=(14, 6))
        plt.plot(ts, Y_real[idx], 'k-', lw=2, label='Actual', alpha=0.9)
        
        for name, pred in results.items():
            c = colors.get(name, 'gray')
            lw = 3 if name == 'PiT-Net' else 1.5
            ls = '-' if name == 'PiT-Net' else '--'
            alpha = 1.0 if name == 'PiT-Net' else 0.6
            plt.plot(ts, pred[idx], color=c, lw=lw, ls=ls, label=name, alpha=alpha)
            
        plt.title(f"Day Case Study (Sample {idx})", fontsize=14)
        plt.ylabel("Power (kW)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        save_path = OUTPUT_DIR / f"benchmark_plot_sample_{idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    run_benchmark()