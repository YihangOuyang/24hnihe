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
DATA_FILE = BASE_DIR / "dataset_ready_for_research.csv"
MODEL_PATH = "transformer_best.pth"
SCALER_PATH =  "scaler.pkl"
PARAM_FILE = BASE_DIR / "physics_params.csv"

RESULT_CSV = BASE_DIR / "final_inference_result_for_origin.csv"
RESULT_IMG = BASE_DIR / "final_inference_plot.png"

TARGET_DATE_STR = "2019-06-22" 

SEQ_LEN = 96
PRED_LEN = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P_RATED = 30.1

LATITUDE = 34.05
LONGITUDE = -118.24
ALTITUDE = 71
# 【核心修改】强制使用固定时区 (UTC-8)，不进行夏令时转换
# 在 pytz 库中，"Etc/GMT+8" 实际上代表 UTC-8 (符号是反的，这是标准定义)
TIMEZONE = "Etc/GMT+8"

FEATURE_COLS = [
    'Target_Power', 'NWP_GHI', 'NWP_DNI', 'NWP_DHI', 
    'NWP_Temp', 'NWP_Wind', 'NWP_Humidity', 'NWP_Cloud', 'NWP_Precip'
]
INPUT_DIM = len(FEATURE_COLS)
# ===========================================

class PhysicalUncertainty:
    def __init__(self, param_file):
        df = pd.read_csv(param_file)
        params = dict(zip(df['parameter'], df['value']))
        self.a, self.beta, self.c = params['a'], params['beta'], params['c']
        
    def get_sigma(self, p_pred):
        p_pu = p_pred / P_RATED
        p_pu_safe = np.maximum(p_pu, 0.05)
        i_f = self.a * np.power(p_pu_safe, self.beta) + self.c
        return i_f * p_pred

def load_model_and_data():
    if not Path(SCALER_PATH).exists():
        raise FileNotFoundError(f"归一化器未找到: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    
    model = TransformerNBEATS(
        num_stacks=2, num_blocks_per_stack=2, input_dim=INPUT_DIM, 
        d_model=64, nhead=4, num_encoder_layers=2, 
        dim_feedforward=128, input_seq_len=SEQ_LEN, output_len=PRED_LEN
    ).to(DEVICE)
    
    model_path_obj = Path(MODEL_PATH)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"模型文件未找到: {MODEL_PATH}")
        
    model.load_state_dict(torch.load(model_path_obj, map_location=DEVICE))
    model.eval()
    
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    
    # 强制统一时区
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(TIMEZONE)
    
    data_raw = df[FEATURE_COLS].values
    data_scaled = scaler.transform(data_raw)
    return model, scaler, data_scaled, df.index

def predict_single_step(model, input_seq_scaled):
    tensor_x = torch.FloatTensor(input_seq_scaled).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(tensor_x)
    return pred_scaled.cpu().numpy().flatten()

def inverse_transform_y(y_scaled, scaler):
    dummy = np.zeros((len(y_scaled), INPUT_DIM))
    dummy[:, 0] = y_scaled
    return scaler.inverse_transform(dummy)[:, 0]

def apply_night_mask(df_in):
    print("   执行物理后处理: 强制夜间置零...")
    
    # 创建副本
    df_result = df_in.copy()
    
    # 1. 解析时间
    times_naive = pd.to_datetime(df_result['Time']).dt.tz_localize(None)
    times_local = times_naive.dt.tz_localize(TIMEZONE)
    
    # 2. 计算太阳高度角
    times_center = pd.DatetimeIndex(times_local) - pd.Timedelta(minutes=7.5)
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz=TIMEZONE)
    solpos = loc.get_solarposition(times_center)
    
    # 3. 生成掩码 (【关键修改】：直接提取 .values 变成 numpy 数组)
    # solpos 带有 DatetimeIndex，而 df_result 带有 RangeIndex
    # 如果不加 .values，Pandas 会试图对齐索引，导致长度翻倍
    mask_elevation = (solpos['elevation'] < -5).values
    
    hours = times_naive.dt.hour
    mask_hour = ((hours >= 22) | (hours <= 4)).values
    
    # 合并掩码 (现在是两个 numpy 数组相加，长度绝对一致)
    final_mask = mask_elevation | mask_hour
    
    # 再次检查长度 (虽然加上 .values 后理论上不会再报错了)
    if len(final_mask) != len(df_result):
        print(f"   [严重错误] 掩码长度({len(final_mask)}) 与 数据长度({len(df_result)}) 依然不一致！")
        # 只有万不得已才截断
        min_len = min(len(final_mask), len(df_result))
        final_mask = final_mask[:min_len]
    
    # 4. 执行置零
    cols_to_zero = ['Pred_24h', 'Lower_24h', 'Upper_24h', 'Sigma_24h',
                    'Pred_4h', 'Lower_4h', 'Upper_4h', 'Sigma_4h']
    
    count = 0
    for col in cols_to_zero:
        if col in df_result.columns:
            # final_mask 现在是 numpy bool 数组，直接用于 loc
            df_result.loc[final_mask, col] = 0.0
            count += 1
            
    print(f"   [完成] 已强制将 {final_mask.sum()} 行数据的 {count} 个列置为 0。")
    
    return df_result

def run_inference():
    print("--- [Step 4] 执行多尺度推断 (24h vs 4h) ---")
    model, scaler, data, time_index = load_model_and_data()
    phys_engine = PhysicalUncertainty(PARAM_FILE)
    
    print(f"正在寻找日期: {TARGET_DATE_STR} ...")
    target_day_mask = (time_index.date == pd.Timestamp(TARGET_DATE_STR).date())
    
    if not any(target_day_mask):
        print(f"错误：找不到 {TARGET_DATE_STR}！")
        return

    indices_of_day = np.where(target_day_mask)[0]
    target_date_idx = indices_of_day[0]
    
    if target_date_idx + 96 > len(data) or target_date_idx < SEQ_LEN:
        print("错误：数据边界不足。")
        return

    target_timestamps = time_index[target_date_idx : target_date_idx + 96]
    print(f"锁定窗口: {target_timestamps[0]} -> {target_timestamps[-1]}")
    
    y_true = inverse_transform_y(data[target_date_idx : target_date_idx + 96, 0], scaler)
    
    # 24h 预测
    input_24h = data[target_date_idx - SEQ_LEN : target_date_idx]
    pred_24h = inverse_transform_y(predict_single_step(model, input_24h), scaler)
    sigma_24h = phys_engine.get_sigma(pred_24h)
    lower_24h = np.maximum(0, pred_24h - 1.96 * sigma_24h)
    upper_24h = np.minimum(P_RATED * 1.2, pred_24h + 1.96 * sigma_24h)
    
    # 4h 预测
    pred_4h_stitched = []
    upper_4h_stitched = []
    
    for i in range(0, 96, 16):
        current_idx = target_date_idx + i
        input_rolling = data[current_idx - SEQ_LEN : current_idx]
        pred_chunk = inverse_transform_y(predict_single_step(model, input_rolling), scaler)
        
        sigma_chunk = phys_engine.get_sigma(pred_chunk)
        upper_chunk = np.minimum(P_RATED * 1.2, pred_chunk + 1.96 * sigma_chunk)
        
        pred_4h_stitched.extend(pred_chunk[:16])
        upper_4h_stitched.extend(upper_chunk[:16])
        
    pred_4h = np.array(pred_4h_stitched)
    sigma_4h_real = (np.array(upper_4h_stitched) - pred_4h) / 1.96
    lower_4h = np.maximum(0, pred_4h - 1.96 * sigma_4h_real)
    upper_4h = np.minimum(P_RATED * 1.2, pred_4h + 1.96 * sigma_4h_real)
    
    # 组装
    df_res = pd.DataFrame({
        'Time': target_timestamps,
        'Actual_Power': y_true,
        'Pred_24h': pred_24h, 'Lower_24h': lower_24h, 'Upper_24h': upper_24h, 'Sigma_24h': sigma_24h,
        'Pred_4h': pred_4h, 'Lower_4h': lower_4h, 'Upper_4h': upper_4h, 'Sigma_4h': sigma_4h_real
    })
    
    # 修复：确保 DataFrame 是全新的，没有索引污染
    df_res = df_res.reset_index(drop=True)
    
    # 执行夜间置零
    df_res = apply_night_mask(df_res)
    
    df_res.to_csv(RESULT_CSV, index=False)
    print(f"数据已导出: {RESULT_CSV}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_res['Time'], df_res['Actual_Power'], 'k-', linewidth=2, label='Actual')
    plt.plot(df_res['Time'], df_res['Pred_24h'], 'b--', label='24h Forecast')
    plt.fill_between(df_res['Time'], df_res['Lower_24h'], df_res['Upper_24h'], color='blue', alpha=0.1)
    plt.plot(df_res['Time'], df_res['Pred_4h'], 'r-', linewidth=2, label='4h Forecast')
    plt.plot(df_res['Time'], df_res['Lower_4h'], 'r:', linewidth=0.5)
    plt.plot(df_res['Time'], df_res['Upper_4h'], 'r:', linewidth=0.5)
    plt.title(f"Unified Forecasting: 24h vs 4h ({TARGET_DATE_STR})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(RESULT_IMG, dpi=150)
    print(f"图表已保存: {RESULT_IMG}")

if __name__ == "__main__":
    run_inference()