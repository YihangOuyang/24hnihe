# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import pvlib
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================
# 1. 地理位置 (Stanford)
LATITUDE = 37.427963
LONGITUDE = -122.154785
ALTITUDE = 30 

# 2. 输入文件路径 (改为读取上一步生成的 Parquet 文件)
NWP_PARQUET_PATH = r"./Stanford_NWP_Processed/NWP_Stanford_2018_2019_Full.parquet"
PV_DATA_PATH = r"rawdata/merged_2018_2019_5min_UTC_cleaned.csv"

# 3. 输出路径
OUTPUT_PATH = r"rawdata/Final_Training_Dataset_15min.csv"

# 4. 目标频率
TARGET_FREQ = '15min'

# 5. 变量映射 (Parquet 中的列名 -> 训练集需要的列名)
NWP_COL_MAP = {
    'NWP_GHI': 'NWP_GHI',
    'NWP_BHI': 'NWP_DNI', 
    'NWP_DHI': 'NWP_DHI',
    'NWP_T2m': 'NWP_Temp',
    'NWP_WS10m': 'NWP_Wind',
    'NWP_RH': 'NWP_Humidity',
    'NWP_LCC': 'NWP_Cloud',
    'NWP_Precip': 'NWP_Precip',
    'NWP_Press': 'NWP_Press'
}
# ===========================================

def get_clearsky_profile(times, lat, lon, alt):
    """计算理论晴空辐射 (Ineichen Model)"""
    location = pvlib.location.Location(lat, lon, altitude=alt)
    # 使用 climatology 里的 Linke Turbidity 会更准，这里用固定值 3.0 也可接受
    cs = location.get_clearsky(times, model='ineichen', linke_turbidity=3)
    return cs

def load_nwp_parquet(file_path):
    """
    [极速读取] 读取整个 Parquet 文件
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"未找到 NWP Parquet 文件: {file_path}。请先运行 Step 1。")
    
    print(f"   [IO] 正在加载 NWP Parquet 文件...")
    df = pd.read_parquet(file_path)
    
    # 确保是 UTC (Parquet 通常保留时区，但双重确认更安全)
    if df.index.tz is None:
        print("   [Warn] Parquet 索引丢失时区信息，假设为 UTC")
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
        
    print(f"   [IO] 加载完成。行数: {len(df)}, 时间范围: {df.index[0]} -> {df.index[-1]}")
    return df

def physics_aware_upsample(df_1h, target_freq='15min'):
    """
    [物理升采样] 利用 Clear Sky Index 进行插值，保留瞬态云层特征
    """
    print(f"   [Physics] 执行 NWP 物理升采样 (1h -> {target_freq})...")
    
    target_index = pd.date_range(
        start=df_1h.index[0], end=df_1h.index[-1], 
        freq=target_freq, tz='UTC'
    )
    
    df_result = pd.DataFrame(index=target_index)
    
    # 计算晴空辐射基准
    cs_1h = get_clearsky_profile(df_1h.index, LATITUDE, LONGITUDE, ALTITUDE)
    cs_target = get_clearsky_profile(target_index, LATITUDE, LONGITUDE, ALTITUDE)
    
    # --- A. 辐射变量: Kc (Clear Sky Index) 插值 ---
    # 你的逻辑非常棒，我只做了微调
    rad_map = {
        'NWP_GHI': 'ghi', # df_col: cs_col
        'NWP_BHI': 'dni', 
        'NWP_DHI': 'dhi'
    }
    
    for df_col, cs_col in rad_map.items():
        if df_col in df_1h.columns:
            # 1. 计算 Kc (避免除零)
            valid_sun = cs_1h[cs_col] > 10.0
            k_series_1h = pd.Series(0.0, index=df_1h.index)
            k_series_1h[valid_sun] = df_1h.loc[valid_sun, df_col] / cs_1h.loc[valid_sun, cs_col]
            
            # 限制 Kc 范围 (物理约束，Kc通常不会超过 2.0)
            k_series_1h = k_series_1h.clip(0, 2.0)
            
            # 2. 线性插值 Kc
            k_series_target = k_series_1h.reindex(target_index).interpolate(method='linear').fillna(0)
            
            # 3. 还原: Radiation = Kc_interp * CS_target
            # 并重命名为最终需要的列名 (如果有映射)
            final_col_name = NWP_COL_MAP.get(df_col, df_col)
            df_result[final_col_name] = (k_series_target * cs_target[cs_col]).clip(lower=0)

    # --- B. 气象变量: Cubic 插值 ---
    # 气温、气压变化平滑，用 Cubic 效果好
    meteo_cols = ['NWP_T2m', 'NWP_WS10m', 'NWP_RH', 'NWP_Press'] 
    for col in meteo_cols:
        if col in df_1h.columns:
            final_col_name = NWP_COL_MAP.get(col, col)
            df_result[final_col_name] = df_1h[col].reindex(target_index).interpolate(method='cubic').ffill().bfill()
            
    # --- C. 阶跃/离散变量: Linear ---
    # 云量、降水不适合 Cubic (会出现负值震荡)
    linear_cols = ['NWP_LCC', 'NWP_Precip']
    for col in linear_cols:
        if col in df_1h.columns:
            final_col_name = NWP_COL_MAP.get(col, col)
            df_result[final_col_name] = df_1h[col].reindex(target_index).interpolate(method='linear').clip(lower=0)
            
    return df_result

def prepare_data():
    print(f"=== [Step 2] 构建融合数据集 (Parquet -> {TARGET_FREQ} Tensor) ===")
    
    # 1. 加载 NWP (Parquet)
    try:
        df_nwp_raw = load_nwp_parquet(NWP_PARQUET_PATH)
    except Exception as e:
        print(f"❌ NWP 加载失败: {e}")
        return

    # 2. 物理升采样
    df_nwp_15min = physics_aware_upsample(df_nwp_raw, target_freq=TARGET_FREQ)

    # 3. 加载 PV 实测数据
    pv_path = Path(PV_DATA_PATH)
    if not pv_path.exists():
        print(f"⚠️ 未找到实测 PV 数据: {pv_path}，仅输出 NWP。")
        df_nwp_15min.to_csv(Path("rawdata") / "Processed_NWP_15min_Only.csv")
        return
        
    print("   [PV] 加载实测数据...")
    df_pv = pd.read_csv(pv_path)
    # 健壮的 Time 列寻找逻辑
    time_cols = [c for c in df_pv.columns if 'time' in c.lower() or 'date' in c.lower()]
    if not time_cols:
        raise ValueError("无法在 PV 数据中找到时间列")
    
    df_pv[time_cols[0]] = pd.to_datetime(df_pv[time_cols[0]], utc=True)
    df_pv.set_index(time_cols[0], inplace=True)
    
    # 降采样 PV (5min -> 15min)
    df_pv_15min = df_pv.resample(TARGET_FREQ, label='left', closed='left').mean()

    # 4. 合并 (Inner Join)
    print("   [Merge] 对齐数据...")
    common_idx = df_pv_15min.index.intersection(df_nwp_15min.index)
    
    if common_idx.empty:
        print("❌ 错误: PV 和 NWP 时间无重叠！")
        return

    final_df = pd.concat([
        df_pv_15min.loc[common_idx],
        df_nwp_15min.loc[common_idx]
    ], axis=1)

    # 5. 夜间清洗 (Final Polish)
    print("   [Clean] 夜间数值清洗...")
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz='UTC')
    solpos = loc.get_solarposition(final_df.index)
    # 使用 -2 度甚至 -5 度作为一个安全的 buffer，避免晨昏时段被误切
    is_night = solpos['elevation'] < -5.0 
    
    # 需要置零的列 (包括 NWP 和 实际功率)
    cols_to_zero = ['NWP_GHI', 'NWP_DNI', 'NWP_DHI', 'Power_Actual', 'Target_Power']
    for c in cols_to_zero:
        if c in final_df.columns:
            final_df.loc[is_night, c] = 0.0

    final_df = final_df.fillna(0)
    
    # 6. 保存
    final_df.to_csv(OUTPUT_PATH)
    print(f"\n[✅ 完成] 训练集已就绪: {OUTPUT_PATH}")
    print(f"   Shape: {final_df.shape}")

if __name__ == "__main__":
    prepare_data()