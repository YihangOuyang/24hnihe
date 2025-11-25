# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import pvlib

# ================= 配置区域 =================
# 1. 地理位置
LATITUDE = 34.05
LONGITUDE = -118.24
ALTITUDE = 71
# 2. 目标时区
TARGET_TZ = "America/Los_Angeles"

# 3. 变量映射字典
NWP_COL_MAP = {
    'shortwave_radiation (W/m²)': 'NWP_GHI',
    'direct_normal_irradiance (W/m²)': 'NWP_DNI',
    'diffuse_radiation (W/m²)': 'NWP_DHI',
    'temperature_2m (°C)': 'NWP_Temp',
    'wind_speed_10m (km/h)': 'NWP_Wind',
    'relative_humidity_2m (%)': 'NWP_Humidity',
    'cloud_cover (%)': 'NWP_Cloud',
    'precipitation (mm)': 'NWP_Precip'
}
# ===========================================

def get_clearsky_profile(times, lat, lon, alt):
    location = pvlib.location.Location(lat, lon, altitude=alt)
    cs = location.get_clearsky(times, model='ineichen')
    return cs

def physics_aware_upsample(df_1h, target_freq='15min'):
    print("   [物理处理] 正在执行基于物理机理的重采样...")
    
    start_time = df_1h.index[0]
    end_time = df_1h.index[-1]
    target_index = pd.date_range(start=start_time, end=end_time, freq=target_freq, tz=df_1h.index.tz)
    
    df_result = pd.DataFrame(index=target_index)
    
    print("      - 计算理论晴空辐射基准...")
    cs_1h = get_clearsky_profile(df_1h.index, LATITUDE, LONGITUDE, ALTITUDE)
    cs_15min = get_clearsky_profile(target_index, LATITUDE, LONGITUDE, ALTITUDE)
    
    # --- A. 辐射类变量：晴空指数插值法 ---
    rad_params = {
        'shortwave_radiation (W/m²)': ('ghi', 'NWP_GHI'),
        'direct_normal_irradiance (W/m²)': ('dni', 'NWP_DNI'),
        'diffuse_radiation (W/m²)': ('dhi', 'NWP_DHI')
    }
    
    for raw_col, (cs_type, new_name) in rad_params.items():
        if raw_col in df_1h.columns:
            k_series_1h = df_1h[raw_col] / cs_1h[cs_type]
            k_series_1h = k_series_1h.fillna(0)
            k_series_1h[np.isinf(k_series_1h)] = 0
            k_series_1h[cs_1h[cs_type] < 5] = 0
            
            k_series_15min = k_series_1h.reindex(target_index).interpolate(method='linear').fillna(0)
            df_result[new_name] = k_series_15min * cs_15min[cs_type]
            df_result[new_name] = df_result[new_name].clip(lower=0)

    # --- B. 气象类变量 ---
    meteo_cols = ['temperature_2m (°C)', 'wind_speed_10m (km/h)', 'relative_humidity_2m (%)']
    for col in meteo_cols:
        if col in df_1h.columns:
            new_name = NWP_COL_MAP.get(col, col)
            df_result[new_name] = df_1h[col].reindex(target_index).interpolate(method='cubic').fillna(method='ffill').fillna(method='bfill')

    # --- C. 其他变量 ---
    linear_cols = ['cloud_cover (%)', 'precipitation (mm)']
    for col in linear_cols:
        if col in df_1h.columns:
            new_name = NWP_COL_MAP.get(col, col)
            df_result[new_name] = df_1h[col].reindex(target_index).interpolate(method='linear').fillna(0)
            
    return df_result

def prepare_data():
    base_dir = Path("outputs/clean") 
    pv_path = base_dir / "merged_pv_and_weather_data.csv"
    nwp_path = base_dir / "NWP.csv"
    output_path = base_dir / "dataset_ready_for_research.csv"

    print(f"--- [Step 1] 数据准备: 物理机理融合 ---")
    print(f"--- 源路径: {base_dir}")
    print(f"--- 目标时区: {TARGET_TZ}")

    # 1. 读取实测数据
    if not pv_path.exists():
        print(f"错误: 找不到 {pv_path}")
        return
    print("1. 处理实测数据 (Ground Truth)...")
    df_pv = pd.read_csv(pv_path)
    time_col = 'Timestamp_Local_UTC-08:00'
    df_pv[time_col] = pd.to_datetime(df_pv[time_col], utc=True).dt.tz_convert(TARGET_TZ)
    df_pv.set_index(time_col, inplace=True)
    df_pv = df_pv[~df_pv.index.duplicated(keep='first')]
    df_pv_15min = df_pv.resample('15min').mean()
    
    df_pv_15min = df_pv_15min.rename(columns={
        'Power_Actual': 'Target_Power',
        'ghi': 'Obs_GHI',
        'temp_air': 'Obs_Temp',
        'wind_speed': 'Obs_Wind'
    })
    cols_to_keep = ['Target_Power', 'Obs_GHI', 'Obs_Temp', 'Obs_Wind']
    df_pv_15min = df_pv_15min[cols_to_keep]

    # 2. 读取 NWP 数据
    if not nwp_path.exists():
        print(f"错误: 找不到 {nwp_path}")
        return
    print("2. 处理 NWP 数据 (Forecast Inputs)...")
    df_nwp = pd.read_csv(nwp_path, skiprows=3)
    
    df_nwp['time'] = pd.to_datetime(df_nwp['time'])
    df_nwp['time'] = df_nwp['time'].dt.tz_localize(TARGET_TZ, ambiguous='NaT', nonexistent='shift_forward')
    df_nwp.set_index('time', inplace=True)
    
    df_nwp = df_nwp[~df_nwp.index.duplicated(keep='first')]
    df_nwp = df_nwp[df_nwp.index.notnull()] 

    df_nwp_15min = physics_aware_upsample(df_nwp, target_freq='15min')

    # 3. 数据合并
    print("3. 合并所有变量...")
    common_index = df_pv_15min.index.intersection(df_nwp_15min.index)
    
    if len(common_index) == 0:
        print("错误: 实测数据与NWP数据时间无重叠！")
        return

    final_df = pd.concat([df_pv_15min.loc[common_index], df_nwp_15min.loc[common_index]], axis=1)
    
    # =========================================================
    # 【核心修改】夜间辐射底噪对齐
    # =========================================================
    print("   [清洗] 执行夜间辐射底噪对齐 (Power=0 & Low GHI -> GHI=0)...")
    
    # 逻辑：只有当 Power=0 且 NWP 辐射本来就很小(<20) 时才置0
    # 这样可以避免把白天的故障（Power=0但GHI很高）错误地把 GHI 置 0
    mask_night_clean = (final_df['Target_Power'] == 0) & (final_df['NWP_GHI'] < 10.0)
    
    for col in ['NWP_GHI', 'NWP_DNI', 'NWP_DHI']:
        if col in final_df.columns:
            final_df.loc[mask_night_clean, col] = 0.0
            
    print(f"      - 已修正 {mask_night_clean.sum()} 个夜间噪声点。")
    # =========================================================

    # 保存
    final_df.to_csv(output_path)
    print(f"\n[成功] 数据集已保存至: {output_path}")
    print(f"数据形状: {final_df.shape}")
    
    # 简单验证
    midnight_val = final_df[final_df.index.hour == 0]['NWP_GHI'].mean()
    print(f"\n[验证] 00:00 时刻 NWP_GHI 均值 (应接近0): {midnight_val:.4f}")

if __name__ == "__main__":
    prepare_data()