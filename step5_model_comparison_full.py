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
import warnings

# 导入你的 Transformer 模型定义
try:
    from step0_model import TransformerNBEATS
except ImportError:
    # 为了让脚本独立运行，如果找不到文件，可以把类定义复制到这里，或者报错
    print("Error: model.py not found.")
    exit()

warnings.filterwarnings("ignore")

# ================= 配置区域 =================
BASE_DIR = Path("outputs/clean")
MODEL_CACHE_DIR = BASE_DIR / "benchmark_models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = BASE_DIR / "dataset_ready_for_research_15min.csv"
TRANSFORMER_PATH = "transformer_best.pth"   
SCALER_PATH = "scaler.pkl"

# 要绘制的日期 (确保该日期在测试集中)
PLOT_SPECIFIC_DATE = "2019-10-05" 

# 是否强制重新训练基准模型 (调试时设为False节省时间，出图时设为True)
FORCE_RETRAIN = False 

SEQ_LEN = 96
PRED_LEN = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P_RATED = 30.1
BATCH_SIZE = 64

LATITUDE = 34.05
LONGITUDE = -118.24
# 绘图时使用的当地时区
TIMEZONE_LOCAL = "America/Los_Angeles" 

RAW_COLS = [
    'Target_Power',
    'NWP_GHI', 'NWP_DNI', 'NWP_DHI',
    'NWP_Temp', 'NWP_Wind', 'NWP_Humidity', 'NWP_Cloud', 'NWP_Precip'
]
FULL_COLS = RAW_COLS + ['Solar_El_sin', 'Solar_Az_sin', 'Solar_Az_cos']
INPUT_DIM = len(FULL_COLS)
# ===========================================

# --- 基准模型定义 ---
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
            nn.Linear(self.flatten_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), 
            nn.Linear(256, output_len)
        )
    def forward(self, x):
        return self.net(x)

# ============== 辅助函数 ==============

def add_solar_features(df):
    # 强制 UTC 索引
    try:
        times_utc = pd.to_datetime(df.index, utc=True)
    except:
        times_utc = pd.to_datetime(df.index).tz_localize('UTC')
    
    # 这里的 solpos 必须用 UTC 计算
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz='UTC')
    solpos = loc.get_solarposition(times_utc)
    
    df['Solar_El_sin'] = np.sin(np.radians(solpos['elevation']))
    df['Solar_Az_sin'] = np.sin(np.radians(solpos['azimuth']))
    df['Solar_Az_cos'] = np.cos(np.radians(solpos['azimuth']))
    
    # 更新索引为 UTC，方便后续统一处理
    df.index = times_utc
    return df

def load_and_prep_data():
    print("--- [Step 1] Loading Data ---")
    df = pd.read_csv(DATA_FILE, index_col=0)
    df = add_solar_features(df) # Add features & ensure UTC index

    data = df[FULL_COLS].values
    
    # 加载训练好的 Scaler (防泄露)
    if not Path(SCALER_PATH).exists():
        raise FileNotFoundError(f"归一化器未找到: {SCALER_PATH}，请先运行训练脚本！")
    scaler = joblib.load(SCALER_PATH)
    data_scaled = scaler.transform(data) 

    # 构造样本
    xs, ys = [], []
    sample_times = [] 
    
    for i in range(len(data_scaled) - SEQ_LEN - PRED_LEN):
        xs.append(data_scaled[i:i + SEQ_LEN, :])                  
        ys.append(data_scaled[i + SEQ_LEN:i + SEQ_LEN + PRED_LEN, 0])
        # 记录预测起始时间 (即 y 的第一个点的时间)
        sample_times.append(df.index[i + SEQ_LEN])

    X = np.array(xs)
    Y = np.array(ys)
    
    # 将时间转换为当地时间，以便筛选 "00:00"
    sample_times = pd.to_datetime(sample_times).tz_convert(TIMEZONE_LOCAL)

    # 【核心修正】严格按照 70/15/15 划分
    # 训练基准模型时，使用前 70% (Train)
    # 测试时，使用后 15% (Test)
    # 中间的 15% (Val) 跳过，或者合并到 Train (这里为了简单，基准模型用前85%训练，或严格用70%)
    # 为了公平对比 Transformer (它只在70%上训练)，基准模型也应该只在 70% 上训练
    
    train_split = int(len(X) * 0.70) 
    test_split_start = int(len(X) * 0.85) # 测试集从 85% 开始
    
    print(f"Dataset Split:")
    print(f"  Train: 0 -> {train_split}")
    print(f"  Test : {test_split_start} -> {len(X)}")

    X_train = X[:train_split]
    Y_train = Y[:train_split]
    
    X_test = X[test_split_start:]
    Y_test = Y[test_split_start:]
    test_times = sample_times[test_split_start:]
    
    return X_train, Y_train, X_test, Y_test, scaler, INPUT_DIM, test_times

# ... (训练函数保持不变，稍微增加 Epoch) ...
def train_or_load_torch_model(model_class, model_name, X_train, Y_train, **model_args):
    save_path = MODEL_CACHE_DIR / f"{model_name}.pth"
    model = model_class(**model_args).to(DEVICE)
    
    if save_path.exists() and not FORCE_RETRAIN:
        print(f"   [Load] Loading {model_name}...")
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        return model
        
    print(f"   [Train] Training {model_name}...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model.train()
    for epoch in range(30): # 增加一点 Epoch 保证收敛
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
        print(f"   [Load] Loading {model_name}...")
        return joblib.load(save_path)
        
    print(f"   [Train] Training {model_name}...")
    # Flatten inputs for sklearn: [N, Seq, Feat] -> [N, Seq*Feat]
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
    # 只反归一化第0列 (Target_Power)
    dummy = np.zeros((y_scaled.size, input_dim))
    dummy[:, 0] = y_scaled.flatten()
    inv = scaler.inverse_transform(dummy)[:, 0]
    return inv.reshape(y_scaled.shape)

def calc_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # NMAE (Normalized by Capacity)
    nmae = mae / P_RATED * 100 
    print(f"   >> {name:<15}: RMSE={rmse:.3f} kW, MAE={mae:.3f} kW, NMAE={nmae:.2f}%, R2={r2:.3f}")
    return {'Model': name, 'RMSE': rmse, 'MAE': mae, 'NMAE (%)': nmae, 'R2': r2}

def apply_night_mask(df_in, target_col='Transformer_Pred'):
    """使用物理规律去除夜间噪声"""
    df_result = df_in.copy()
    
    # 1. 恢复时间索引
    times_naive = pd.to_datetime(df_result['Time']).dt.tz_localize(None)
    times_local = times_naive.dt.tz_localize(TIMEZONE_LOCAL)
    # 中心化
    times_center = pd.DatetimeIndex(times_local) - pd.Timedelta(minutes=7.5)
    
    # 2. 计算太阳高度角
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz=TIMEZONE_LOCAL)
    solpos = loc.get_solarposition(times_center)
    
    # 3. 掩码条件：高度角 < -2度 或 时间在 22:00-04:00 之间
    mask_elevation = solpos['elevation'] < -2.0
    hours = times_naive.dt.hour
    mask_hour = (hours >= 22) | (hours <= 4)
    
    final_mask = (mask_elevation | mask_hour).values
    
    if target_col in df_result.columns:
        df_result.loc[final_mask, target_col] = 0.0
            
    return df_result[target_col].values

# ============== 主流程 ==============

def run_full_comparison():
    print("\n--- [Step 5] Model Benchmark (Strict Day-Ahead Test) ---")

    # 1. 数据准备
    X_train, Y_train, X_test_all, Y_test_all, scaler, input_dim, test_times = load_and_prep_data()
    
    # 2. 筛选 "00:00" 起始的样本 (Day-Ahead Logic)
    # test_times 是 DatetimeIndex，直接访问 .hour (不需要 .dt)
    mask_midnight = (test_times.hour == 0) & (test_times.minute == 0)
    num_samples = mask_midnight.sum()
    
    if num_samples == 0:
        print("❌ 错误：测试集中没有找到 00:00 起始的样本！请检查时间范围。")
        return
    else:
        print(f"   [筛选] 保留 00:00 起报样本: {num_samples} 天")
        X_test = X_test_all[mask_midnight]
        Y_test = Y_test_all[mask_midnight]
        test_times_filtered = test_times[mask_midnight]

    # 获取 Ground Truth (反归一化)
    Y_true = inverse_transform_y(Y_test, scaler, input_dim)
    predictions = {} 

    # --- 模型推断 ---
    print("\n1. Benchmarking Persistence...")
    predictions['Persistence'] = np.maximum(0, inverse_transform_y(X_test[:, :, 0], scaler, input_dim))

    print("2. Benchmarking XGBoost...")
    xgb_base = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=6, 
        device="cuda", tree_method="hist", n_jobs=-1
    )
    xgb_model = train_or_load_sklearn_model(MultiOutputRegressor(xgb_base), "xgboost_benchmark", X_train, Y_train)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    predictions['XGBoost'] = np.maximum(0, inverse_transform_y(xgb_model.predict(X_test_flat), scaler, input_dim))

    print("3. Benchmarking MLP...")
    mlp = train_or_load_torch_model(MLPModel, "mlp_benchmark", X_train, Y_train, input_len=SEQ_LEN, input_dim=input_dim, output_len=PRED_LEN)
    predictions['MLP'] = np.maximum(0, inverse_transform_y(predict_torch(mlp, X_test), scaler, input_dim))

    print("4. Benchmarking LSTM...")
    lstm = train_or_load_torch_model(LSTMModel, "lstm_benchmark", X_train, Y_train, input_dim=input_dim, hidden_dim=64, output_len=PRED_LEN)
    predictions['LSTM'] = np.maximum(0, inverse_transform_y(predict_torch(lstm, X_test), scaler, input_dim))

    print("5. Benchmarking Transformer (Ours)...")
    transformer = TransformerNBEATS(
        num_stacks=1, num_blocks_per_stack=1, input_dim=input_dim,
        d_model=32, nhead=4, num_encoder_layers=2,
        dim_feedforward=128, input_seq_len=SEQ_LEN, output_len=PRED_LEN, dropout=0.3
    ).to(DEVICE)
    
    if Path(TRANSFORMER_PATH).exists():
        transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=DEVICE))
        transformer.eval()
        y_trans_raw = inverse_transform_y(predict_torch(transformer, X_test), scaler, input_dim)
        predictions['Transformer'] = np.maximum(0, y_trans_raw)
    else:
        print(f"❌ 警告: 未找到 {TRANSFORMER_PATH}，跳过 Transformer 评估。")

    # --- 计算指标 ---
    print("\n[Global Metrics]")
    results = []
    for name, y_pred in predictions.items():
        results.append(calc_metrics(Y_true.flatten(), y_pred.flatten(), name))
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(BASE_DIR / "final_benchmark_metrics.csv", index=False)
    print(f"指标已保存至 {BASE_DIR / 'final_benchmark_metrics.csv'}")

    # ==========================================
    # 绘图 1: 分步长 RMSE (Step-wise RMSE)
    # ==========================================
    print("\n绘制 Step-wise RMSE...")
    plt.figure(figsize=(10, 6))
    
    steps = np.arange(1, 97)
    colors = {'Persistence': '#7f7f7f', 'XGBoost': '#2ca02c', 'MLP': '#ff7f0e', 'LSTM': '#1f77b4', 'Transformer': '#d62728'}
    styles = {'Persistence': '--', 'XGBoost': ':', 'MLP': '-.', 'LSTM': '-.', 'Transformer': '-'}
    
    for name, y_pred in predictions.items():
        mse_per_step = np.mean((Y_true - y_pred) ** 2, axis=0)
        rmse_per_step = np.sqrt(mse_per_step)
        
        plt.plot(steps, rmse_per_step, label=name, 
                 color=colors.get(name, 'black'), 
                 linestyle=styles.get(name, '-'),
                 linewidth=2.5 if name == 'Transformer' else 1.5)

    plt.title("24-Hour Forecast Error Distribution (Step-wise RMSE)")
    plt.xlabel("Time of Day")
    plt.ylabel("RMSE (kW)")
    plt.xticks([0, 24, 48, 72, 96], ['00:00', '06:00', '12:00', '18:00', '24:00'])
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.savefig(BASE_DIR / "stepwise_rmse_comparison.png", dpi=300)

    # ==========================================
    # 绘图 2: 指定日期波形
    # ==========================================
    print(f"\n绘制日期波形: {PLOT_SPECIFIC_DATE}")
    
    target_date = pd.Timestamp(PLOT_SPECIFIC_DATE).date()
    
    # 【核心修复 1】DatetimeIndex 直接使用 .date，不需要 .dt
    date_mask = (test_times_filtered.date == target_date)
    
    if not date_mask.any():
        print(f"⚠️ 警告: 测试集中没有 {PLOT_SPECIFIC_DATE} 的 00:00 起报数据！")
        idx = 0
        # 【核心修复 2】DatetimeIndex 直接索引，不需要 .iloc
        target_date = test_times_filtered[0].date() 
        print(f"   -> 自动切换到测试集第一天: {target_date}")
    else:
        idx = np.where(date_mask)[0][0]

    # 【核心修复 3】使用直接索引 [] 而不是 .iloc[]
    t_start = test_times_filtered[idx]
    
    t_axis = pd.date_range(start=t_start, periods=96, freq='15min')
    
    plt.figure(figsize=(12, 6))
    
    # Ground Truth
    plt.plot(t_axis, Y_true[idx], 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
    
    if 'Persistence' in predictions:
        plt.plot(t_axis, predictions['Persistence'][idx], color=colors['Persistence'], linestyle='--', alpha=0.6, label='Persistence')
    
    if 'XGBoost' in predictions:
        plt.plot(t_axis, predictions['XGBoost'][idx], color=colors['XGBoost'], linestyle=':', linewidth=1.5, label='XGBoost')
        
    if 'LSTM' in predictions:
        plt.plot(t_axis, predictions['LSTM'][idx], color=colors['LSTM'], linestyle='-.', linewidth=1.5, label='LSTM')
        
    if 'Transformer' in predictions:
        raw_pred = predictions['Transformer'][idx]
        temp_df = pd.DataFrame({'Time': t_axis, 'Transformer_Pred': raw_pred})
        clean_pred = apply_night_mask(temp_df) 
        plt.plot(t_axis, clean_pred, color=colors['Transformer'], linewidth=3, label='Transformer (Ours)')

    plt.title(f"Day-Ahead Forecasting: {target_date}")
    plt.ylabel("Power (kW)")
    plt.xlabel("Local Time")
    
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(BASE_DIR / "waveform_comparison.png", dpi=300)
    print(f"图片已保存至 {BASE_DIR}")

if __name__ == "__main__":
    run_full_comparison()