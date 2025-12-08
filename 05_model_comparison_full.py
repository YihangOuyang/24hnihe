# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import joblib
from pathlib import Path
import pvlib

from model import TransformerNBEATS

# ================= 配置区域 =================
BASE_DIR = Path("outputs/clean")
MODEL_CACHE_DIR = BASE_DIR / "benchmark_models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = BASE_DIR / "dataset_ready_for_research_15min.csv"
TRANSFORMER_PATH = "transformer_best.pth"   
SCALER_PATH = "scaler.pkl"
PARAM_FILE = BASE_DIR / "physics_params.csv" # 如果需要物理参数计算sigma

PLOT_SPECIFIC_DATE = "2019-10-05" 
FORCE_RETRAIN = True 

SEQ_LEN = 96
PRED_LEN = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P_RATED = 30.1
BATCH_SIZE = 64

LATITUDE = 34.05
LONGITUDE = -118.24
ALTITUDE = 71
TIMEZONE = "UTC"

RAW_COLS = [
    'Target_Power',
    'NWP_GHI', 'NWP_DNI', 'NWP_DHI',
    'NWP_Temp', 'NWP_Wind', 'NWP_Humidity', 'NWP_Cloud', 'NWP_Precip'
]
FULL_COLS = RAW_COLS + ['Solar_El_sin', 'Solar_Az_sin', 'Solar_Az_cos']
INPUT_DIM = len(FULL_COLS)
# ===========================================

# ... (LSTM, MLP 类定义保持不变) ...
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
            nn.Flatten(), nn.Linear(self.flatten_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, output_len)
        )
    def forward(self, x):
        return self.net(x)

# ============== 辅助函数 ==============

def add_solar_features(df):
    try:
        times = pd.to_datetime(df.index, utc=True).tz_convert(TIMEZONE)
    except:
        times = pd.to_datetime(df.index).tz_localize(TIMEZONE)
    
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz=TIMEZONE)
    solpos = loc.get_solarposition(times)
    
    df['Solar_El_sin'] = np.sin(np.radians(solpos['elevation']))
    df['Solar_Az_sin'] = np.sin(np.radians(solpos['azimuth']))
    df['Solar_Az_cos'] = np.cos(np.radians(solpos['azimuth']))
    return df

def load_and_prep_data():
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    df = add_solar_features(df)

    data = df[FULL_COLS].values
    if not Path(SCALER_PATH).exists():
        raise FileNotFoundError(f"归一化器未找到: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    data_scaled = scaler.transform(data) 

    xs, ys = [], []
    sample_times = [] 
    
    for i in range(len(data_scaled) - SEQ_LEN - PRED_LEN):
        xs.append(data_scaled[i:i + SEQ_LEN, :])                  
        ys.append(data_scaled[i + SEQ_LEN:i + SEQ_LEN + PRED_LEN, 0])
        sample_times.append(df.index[i + SEQ_LEN])

    X = np.array(xs)
    Y = np.array(ys)
    sample_times = pd.to_datetime(sample_times, utc=True).tz_convert(TIMEZONE)

    split = int(len(X) * 0.8)
    return X[:split], Y[:split], X[split:], Y[split:], scaler, INPUT_DIM, sample_times[split:]

# ... (模型训练/加载函数保持不变) ...
def train_or_load_torch_model(model_class, model_name, X_train, Y_train, **model_args):
    save_path = MODEL_CACHE_DIR / f"{model_name}.pth"
    model = model_class(**model_args).to(DEVICE)
    if save_path.exists() and not FORCE_RETRAIN:
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        return model
    print(f"   [Train] Training {model_name}...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for epoch in range(15):
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), save_path)
    return model

def train_or_load_sklearn_model(model_obj, model_name, X_train, Y_train):
    save_path = MODEL_CACHE_DIR / f"{model_name}.pkl"
    if save_path.exists() and not FORCE_RETRAIN:
        return joblib.load(save_path)
    print(f"   [Train] Training {model_name}...")
    X_flat = X_train.reshape(X_train.shape[0], -1)
    model_obj.fit(X_flat, Y_train)
    joblib.dump(model_obj, save_path)
    return model_obj

def predict_torch(model, X_test):
    model.eval()
    preds = []
    dataset = TensorDataset(torch.FloatTensor(X_test))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for bx, in loader:
            bx = bx.to(DEVICE)
            out = model(bx)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)

def inverse_transform_y(y_scaled, scaler, input_dim):
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

# =========================================================
# 【核心移植】来自 04 的防弹版夜间掩码函数
# =========================================================
def apply_night_mask(df_in):
    # print("   执行物理后处理...") 
    # (为了防止 benchmark 刷屏，这里注释掉 print，或者保留)
    df_result = df_in.copy()
    n_rows = len(df_result)
    
    times_naive = pd.to_datetime(df_result['Time']).dt.tz_localize(None)
    times_local = times_naive.dt.tz_localize(TIMEZONE)
    times_center = pd.DatetimeIndex(times_local) - pd.Timedelta(minutes=7.5)
    
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz=TIMEZONE)
    solpos = loc.get_solarposition(times_center)
    
    mask_elevation = solpos['elevation'] < -2.0
    hours = times_naive.dt.hour
    mask_hour = (hours >= 22) | (hours <= 4)
    
    final_mask = (mask_elevation | mask_hour).values
    
    if len(final_mask) != n_rows:
        final_mask = final_mask[:n_rows]
    
    # 对指定列置0
    cols_to_zero = ['Transformer_Pred'] 
    for col in cols_to_zero:
        if col in df_result.columns:
            df_result.loc[final_mask, col] = 0.0
            
    return df_result['Transformer_Pred'].values

# ============== 4. 主程序：全面模型对比 ==============

def run_full_comparison():
    print("--- [Step 5] 全面模型对比实验 (Strict Day-Ahead + Step-wise RMSE) ---")

    # 1. 数据加载
    X_train, Y_train, X_test_all, Y_test_all, scaler, input_dim, test_times = load_and_prep_data()
    
    # 2. 筛选 00:00 起报
    mask_midnight = (test_times.hour == 0) & (test_times.minute == 0)
    num_samples = mask_midnight.sum()
    
    if num_samples == 0:
        print("警告：测试集中没有找到 00:00 起始的样本！")
        X_test = X_test_all
        Y_test = Y_test_all
        test_times_filtered = test_times
    else:
        print(f"   [筛选] 仅保留日前预测样本 (00:00 Start): {num_samples} 个天")
        X_test = X_test_all[mask_midnight]
        Y_test = Y_test_all[mask_midnight]
        test_times_filtered = test_times[mask_midnight]

    Y_true = inverse_transform_y(Y_test, scaler, input_dim)
    predictions = {} 

    # --- 模型推断 (统一加上非负约束) ---
    print("1. Running Persistence...")
    predictions['Persistence'] = np.maximum(0, inverse_transform_y(X_test[:, :, 0], scaler, input_dim))

    print("2. Running XGBoost...")
    # 修改后 (启用 GPU 加速)
    xgb_base = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        device="cuda",  # 或者 tree_method='gpu_hist' (旧版本写法)
        n_jobs=-1
    )
    xgb_model = train_or_load_sklearn_model(MultiOutputRegressor(xgb_base), "xgboost_benchmark", X_train, Y_train)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    predictions['XGBoost'] = np.maximum(0, inverse_transform_y(xgb_model.predict(X_test_flat), scaler, input_dim))

    print("3. Running MLP...")
    mlp = train_or_load_torch_model(MLPModel, "mlp_benchmark", X_train, Y_train, input_len=SEQ_LEN, input_dim=input_dim, output_len=PRED_LEN)
    predictions['MLP'] = np.maximum(0, inverse_transform_y(predict_torch(mlp, X_test), scaler, input_dim))

    print("4. Running LSTM...")
    lstm = train_or_load_torch_model(LSTMModel, "lstm_benchmark", X_train, Y_train, input_dim=input_dim, hidden_dim=64, output_len=PRED_LEN)
    predictions['LSTM'] = np.maximum(0, inverse_transform_y(predict_torch(lstm, X_test), scaler, input_dim))

    print("5. Running Transformer (Ours)...")
    transformer = TransformerNBEATS(
        num_stacks=1, num_blocks_per_stack=1, input_dim=input_dim,
        d_model=32, nhead=4, num_encoder_layers=2,
        dim_feedforward=128, input_seq_len=SEQ_LEN, output_len=PRED_LEN, dropout=0.3
    ).to(DEVICE)
    
    if Path(TRANSFORMER_PATH).exists():
        transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=DEVICE))
    else:
        print("警告: 未找到 Transformer 模型！")
    
    y_trans_raw = inverse_transform_y(predict_torch(transformer, X_test), scaler, input_dim)
    predictions['Transformer'] = np.maximum(0, y_trans_raw)

    # --- 计算全局指标 ---
    results = []
    for name, y_pred in predictions.items():
        results.append(calc_metrics(Y_true.flatten(), y_pred.flatten(), name))
    pd.DataFrame(results).to_csv(BASE_DIR / "final_benchmark_results.csv", index=False)

    # ==========================================
    # 【核心新增】绘图 1: 分步长 RMSE 曲线
    # ==========================================
    print("\n绘制分步长 RMSE 曲线...")
    plt.figure(figsize=(10, 6))
    
    # 步长 (1~96)
    steps = np.arange(1, 97)
    # X轴刻度标签 (每6小时一个标记)
    step_ticks = [0, 24, 48, 72, 96]
    step_labels = ['00:00', '06:00', '12:00', '18:00', '24:00']
    
    colors = {'Persistence': 'gray', 'XGBoost': 'green', 'MLP': 'orange', 'LSTM': 'blue', 'Transformer': 'red'}
    
    for name, y_pred in predictions.items():
        # y_pred: [N_samples, 96]
        # 对 axis=0 求 MSE -> 得到 96 个 MSE 值
        mse_per_step = np.mean((Y_true - y_pred) ** 2, axis=0)
        rmse_per_step = np.sqrt(mse_per_step)
        
        plt.plot(steps, rmse_per_step, label=name, color=colors.get(name, 'black'), 
                 linewidth=2.5 if name == 'Transformer' else 1.5)

    plt.title("Step-wise RMSE Evaluation (24h Horizon)")
    plt.xlabel("Time of Day")
    plt.ylabel("RMSE (kW)")
    plt.xticks(step_ticks, step_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(BASE_DIR / "stepwise_rmse.png", dpi=150)
    print(f"分步长误差图已保存: {BASE_DIR / 'stepwise_rmse.png'}")

    # ==========================================
    # 绘图 2: 指定日期波形 (含严格后处理)
    # ==========================================
    print(f"\n绘制指定日期 ({PLOT_SPECIFIC_DATE}) 的波形...")
    
    date_mask = (test_times_filtered.date == pd.Timestamp(PLOT_SPECIFIC_DATE).date())
    
    if not any(date_mask):
        print(f"警告：测试集中找不到 {PLOT_SPECIFIC_DATE}！")
        idx = np.random.randint(0, len(Y_test))
        title_date = str(test_times_filtered[idx].date())
    else:
        idx = np.where(date_mask)[0][0]
        title_date = PLOT_SPECIFIC_DATE

    # 获取 Transformer 当日数据并做夜间清洗
    y_trans_plot = predictions['Transformer'][idx]
    
    plot_start_time = test_times_filtered[idx]
    plot_timestamps = pd.date_range(start=plot_start_time, periods=96, freq='15min')
    
    temp_df = pd.DataFrame({'Time': plot_timestamps, 'Transformer_Pred': y_trans_plot})
    y_trans_clean = apply_night_mask(temp_df)

    plt.figure(figsize=(14, 7))
    plt.plot(Y_true[idx], 'k-', linewidth=3, label='Ground Truth')
    plt.plot(predictions['Persistence'][idx], color='gray', linestyle='--', alpha=0.5, label='Persistence')
    plt.plot(predictions['XGBoost'][idx], color='green', linestyle=':', label='XGBoost')
    plt.plot(predictions['LSTM'][idx], color='blue', linestyle='-.', label='LSTM')
    
    # 绘制清洗后的 Transformer
    plt.plot(y_trans_clean, color='red', linewidth=2.5, label='Transformer (Ours)')

    plt.title(f"Day-Ahead Forecast Comparison ({title_date})")
    plt.xticks(ticks=[0, 24, 48, 72, 96], labels=['00:00', '06:00', '12:00', '18:00', '00:00'])
    plt.xlabel("Time of Day")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(BASE_DIR / "full_model_comparison.png")
    print(f"单日对比图已保存: {BASE_DIR / 'full_model_comparison.png'}")

if __name__ == "__main__":
    run_full_comparison()