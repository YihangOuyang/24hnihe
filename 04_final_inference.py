# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib
import pvlib
from pathlib import Path
from model import TransformerNBEATS

# ================= 配置区域 =================
BASE_DIR = Path("outputs/clean")
DATA_FILE = BASE_DIR / "dataset_ready_for_research_15min.csv"
MODEL_PATH = "transformer_best.pth"
SCALER_PATH = "scaler.pkl"
PARAM_FILE = BASE_DIR / "physics_params.csv"

RESULT_CSV = BASE_DIR / "final_inference_result_test_set.csv"
RESULT_IMG = BASE_DIR / "final_inference_plot_test_set.png"

# 【重要】要预测的日期
PREFERRED_DATE_STR = "2019-10-05" 

SEQ_LEN = 96          
PRED_LEN = 96         
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P_RATED = 30.1        

# 站点参数
LATITUDE = 34.05
LONGITUDE = -118.24
ALTITUDE = 71
# TIMEZONE = "Etc/GMT+8"
# TIMEZONE = "America/Los_Angeles"
TIMEZONE = "UTC"
RAW_COLS = [
    'Target_Power',
    'NWP_GHI', 'NWP_DNI', 'NWP_DHI',
    'NWP_Temp', 'NWP_Wind', 'NWP_Humidity', 'NWP_Cloud', 'NWP_Precip'
]
# ===========================================

class PhysicalUncertainty:
    def __init__(self, param_file):
        if not param_file.exists():
            print(f"警告: 物理参数文件 {param_file} 不存在，使用默认参数。")
            self.a, self.beta, self.c = 0.1, 1.0, 0.01 
        else:
            df = pd.read_csv(param_file)
            params = dict(zip(df['parameter'], df['value']))
            self.a, self.beta, self.c = params['a'], params['beta'], params['c']

    def get_sigma(self, p_pred):
        p_safe = np.maximum(p_pred, 0.0)
        p_pu = p_safe / P_RATED
        p_pu_clamped = np.maximum(p_pu, 0.05)
        i_f = self.a * np.power(p_pu_clamped, self.beta) + self.c
        return i_f * p_safe

# 【核心修复 1】更稳健的特征添加与时间索引处理
def add_solar_features(df):
    print("   [预处理] 生成太阳几何特征...")
    
    # 1. 强制统一时间索引 (解决 utc=True 报错的关键)
    # 无论原始索引是字符串、naive datetime 还是 mixed aware，全部强制转为 UTC
    try:
        # utc=True 是解决报错的核心：它告诉 pandas "不管输入长啥样，都给我转成 UTC"
        times_utc = pd.to_datetime(df.index, utc=True)
        # 然后再统一转到目标时区
        times_local = times_utc.tz_convert(TIMEZONE)
    except Exception as e:
        print(f"   [Error] 时间索引转换失败: {e}")
        raise e
    
    # 更新 df 的索引，确保后续操作使用的是处理好的 DatetimeIndex
    df.index = times_local

    # 2. 计算太阳位置
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz=TIMEZONE)
    # 使用更新后的 df.index
    solpos = loc.get_solarposition(df.index)
    
    df['Solar_El_sin'] = np.sin(np.radians(solpos['elevation']))
    df['Solar_Az_sin'] = np.sin(np.radians(solpos['azimuth']))
    df['Solar_Az_cos'] = np.cos(np.radians(solpos['azimuth']))
    return df

def load_model_and_data():
    FULL_COLS = RAW_COLS + ['Solar_El_sin', 'Solar_Az_sin', 'Solar_Az_cos']
    INPUT_DIM = len(FULL_COLS)

    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    df = add_solar_features(df) # 这里会更新 df.index 为正确的时区格式
    
    if not Path(SCALER_PATH).exists():
        raise FileNotFoundError("Scaler not found. Run training first.")
    scaler = joblib.load(SCALER_PATH)
    
    data_raw = df[FULL_COLS].values
    data_scaled = scaler.transform(data_raw)
    
    model = TransformerNBEATS(
        num_stacks=1, num_blocks_per_stack=1, input_dim=INPUT_DIM,
        d_model=32, nhead=4, num_encoder_layers=2,
        dim_feedforward=128, input_seq_len=SEQ_LEN, output_len=PRED_LEN,
        dropout=0.3
    ).to(DEVICE)
    
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError("Model weights not found. Run training first.")
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 返回 df.index，此时它已经是 clean 的 DatetimeIndex
    return model, scaler, data_scaled, df.index, INPUT_DIM

def predict_single_step(model, input_seq_scaled):
    tensor_x = torch.FloatTensor(input_seq_scaled).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(tensor_x)
    return pred_scaled.cpu().numpy().flatten()

def inverse_transform_y(y_scaled, scaler, input_dim):
    dummy = np.zeros((len(y_scaled), input_dim))
    dummy[:, 0] = y_scaled 
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]

def apply_night_mask(df_in):
    df = df_in.copy()
    
    # 这里不需要再做复杂的 to_datetime，因为 Time 列已经是 datetime 类型
    # 如果不是，保险起见再转一次，带上 utc=True
    times = pd.to_datetime(df["Time"], utc=True).dt.tz_convert(TIMEZONE)
        
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz=TIMEZONE, altitude=ALTITUDE)
    solpos = loc.get_solarposition(times)
    
    night_mask = (solpos["apparent_elevation"] < -15.0).to_numpy()
    cols_check = [c for c in df.columns if any(x in c for x in ['Pred', 'Lower', 'Upper', 'Sigma'])]
    
    for col in cols_check:
        df.loc[night_mask, col] = 0.0
        
    return df

def run_inference():
    print("--- [Step 4] 执行多尺度推断 (Strict Test Set Only) ---")
    model, scaler, data, time_index, input_dim = load_model_and_data()
    phys_engine = PhysicalUncertainty(PARAM_FILE)
    
    # ================= 关键逻辑：确定测试集范围 =================
    total_len = len(data)
    test_start_idx = int(total_len * 0.85) 
    
    # 【修复 1】为了准确找到用户想要的“当地日期”，先创建一个临时的当地时间索引
    # time_index 目前是 UTC。我们转为 LA 时间仅用于搜索。
    time_index_la = time_index.tz_convert('America/Los_Angeles')
    
    test_start_date = time_index_la[test_start_idx].date()
    test_end_date = time_index_la[-1].date()
    
    print(f"\n[数据划分信息 (Local Time)]")
    print(f"   Total Samples: {total_len}")
    print(f"   Test Set Start Index: {test_start_idx}")
    print(f"   Test Set Date Range: {test_start_date} ~ {test_end_date}")
    
    # 获取用户想要日期的 date 对象
    preferred_date = pd.Timestamp(PREFERRED_DATE_STR).date()
    
    # 在“当地时间”索引中搜索这个日期
    local_dates = time_index_la.date 
    indices = np.where(local_dates == preferred_date)[0]
    
    target_idx = -1
    if len(indices) == 0:
        print(f"\n[警告] 数据集中找不到当地日期 {PREFERRED_DATE_STR}")
        target_idx = test_start_idx + SEQ_LEN 
        print(f"-> 自动切换到测试集第一天: {local_dates[target_idx]}")
    else:
        # 找到当天的起始点（比如凌晨 00:00 对应的索引）
        # 我们希望预测从这一天的早晨开始，所以取 indices[0]
        # 但是要注意 sequence 长度，不能太靠前
        idx_candidate = indices[0]
        
        # 如果这一天正好是测试集开头，可能前面没有足够的 SEQ_LEN 做输入
        if idx_candidate < SEQ_LEN:
             idx_candidate = SEQ_LEN

        if idx_candidate < test_start_idx:
            print(f"\n[警告] 日期 {PREFERRED_DATE_STR} 位于训练/验证集。")
            # 这里的逻辑您可以根据需要调整，是强制切到测试集，还是允许画训练集
            # 这里保持原逻辑：强制切到测试集
            target_idx = test_start_idx + SEQ_LEN 
            print(f"-> 自动切换到测试集第一天: {local_dates[target_idx]}")
        else:
            print(f"\n[确认] 找到当地日期 {PREFERRED_DATE_STR}，起始索引: {idx_candidate}")
            target_idx = idx_candidate

    # 双重边界检查
    if target_idx < SEQ_LEN or target_idx + PRED_LEN > total_len:
        print("错误：选定日期的索引超出边界。")
        return

    # 锁定 UTC 时间窗口用于模型输入 (Model 输入必须是 UTC 数据)
    # 注意：target_idx 是我们在当地时间轴上找到的位置，它在 UTC 轴上是一样的位置
    target_timestamps_utc = time_index[target_idx : target_idx + PRED_LEN]
    print(f"执行推断窗口 (UTC): {target_timestamps_utc[0]} -> {target_timestamps_utc[-1]}")
    
    # ================= 开始预测 (核心计算保持 UTC) =================
    y_true_scaled = data[target_idx : target_idx + PRED_LEN, 0]
    y_true = inverse_transform_y(y_true_scaled, scaler, input_dim)
    
    # 24h Forecast
    input_24h = data[target_idx - SEQ_LEN : target_idx]
    pred_24h_scaled = predict_single_step(model, input_24h)
    pred_24h = inverse_transform_y(pred_24h_scaled, scaler, input_dim)
    
    pred_24h = np.maximum(0, pred_24h)
    sigma_24h = phys_engine.get_sigma(pred_24h)
    lower_24h = np.maximum(0, pred_24h - 1.96 * sigma_24h)
    upper_24h = np.minimum(P_RATED * 1.2, pred_24h + 1.96 * sigma_24h)
    
    # 4h Rolling Forecast
    pred_4h_list = []
    upper_4h_list = []
    
    step_size = 16 
    for i in range(0, PRED_LEN, step_size):
        curr = target_idx + i
        input_roll = data[curr - SEQ_LEN : curr]
        
        p_chunk_scaled = predict_single_step(model, input_roll)
        p_chunk = inverse_transform_y(p_chunk_scaled, scaler, input_dim)
        p_chunk = np.maximum(0, p_chunk) 
        
        s_chunk = phys_engine.get_sigma(p_chunk)
        u_chunk = np.minimum(P_RATED * 1.2, p_chunk + 1.96 * s_chunk)
        
        end_slice = min(step_size, PRED_LEN - i)
        pred_4h_list.extend(p_chunk[:end_slice])
        upper_4h_list.extend(u_chunk[:end_slice])
        
    pred_4h = np.array(pred_4h_list)
    upper_4h_raw = np.array(upper_4h_list)
    
    sigma_4h_derived = (upper_4h_raw - pred_4h) / 1.96
    lower_4h = np.maximum(0, pred_4h - 1.96 * sigma_4h_derived)
    upper_4h = upper_4h_raw

    # ================= 结果导出与绘图优化 =================
    df_res = pd.DataFrame({
        'Time': target_timestamps_utc, # 此时还是 UTC
        'Actual_Power': y_true,
        'Pred_24h': pred_24h, 'Lower_24h': lower_24h, 'Upper_24h': upper_24h,
        'Pred_4h': pred_4h,   'Lower_4h': lower_4h,   'Upper_4h': upper_4h
    })
    
    df_res = apply_night_mask(df_res)
    
    # 2. [可视化专用] 转换为洛杉矶时间 并 去除时区信息
    df_plot = df_res.copy()
    
    # 【修复 2】强制转换为当地时间，然后 strip (tz_localize(None))
    # 这一步能保证图表上的时间绝对是 00:00 - 23:00，没有时区偏移干扰
    df_plot['Time'] = df_plot['Time'].dt.tz_convert('America/Los_Angeles').dt.tz_localize(None)
    
    # 打印一下看看时间是否对劲
    print(f"\n[绘图时间检查] 起始: {df_plot['Time'].iloc[0]} | 结束: {df_plot['Time'].iloc[-1]}")
    
    df_plot.to_csv(RESULT_CSV, index=False)
    print(f"CSV 已保存 (Local Time): {RESULT_CSV}")
    
    # 3. 绘图
    plt.figure(figsize=(14, 7))
    
    plt.plot(df_plot['Time'], df_plot['Actual_Power'], 'k-', linewidth=2.5, label='Ground Truth')
    
    plt.plot(df_plot['Time'], df_plot['Pred_24h'], color='blue', linestyle='--', linewidth=2, label='Day-Ahead (24h)')
    plt.fill_between(df_plot['Time'], df_plot['Lower_24h'], df_plot['Upper_24h'], color='blue', alpha=0.1, label='95% CI (24h)')
    
    plt.plot(df_plot['Time'], df_plot['Pred_4h'], color='red', linestyle='-', linewidth=2, label='Intra-Day (4h Rolling)')
    plt.plot(df_plot['Time'], df_plot['Lower_4h'], color='red', linestyle=':', linewidth=0.5, alpha=0.7)
    plt.plot(df_plot['Time'], df_plot['Upper_4h'], color='red', linestyle=':', linewidth=0.5, alpha=0.7)
    
    # 获取标题用的日期字符串
    title_date = df_plot['Time'].iloc[0].date()
    
    plt.title(f"Forecast Comparison: {title_date} (Los Angeles Time)\n(Day-Ahead vs. Intra-Day Rolling)", fontsize=14)
    plt.ylabel("Power (kW)", fontsize=12)
    plt.xlabel("Local Time", fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # 格式化 X 轴 (因为已经去除了时区，这里会显示干净的 HH:MM)
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(RESULT_IMG, dpi=150)
    print(f"Plot 已保存: {RESULT_IMG}")

if __name__ == "__main__":
    run_inference()