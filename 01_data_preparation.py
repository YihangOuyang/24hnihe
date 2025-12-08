# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import pvlib
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================
# 1. 地理位置 (Los Angeles)
LATITUDE = 34.05
LONGITUDE = -118.24
ALTITUDE = 71

# 2. [关键修改] 目标时区设为 UTC
# 因为你的 PV 数据已经是 UTC，我们将 NWP 也转到 UTC，统一标准
TARGET_TZ = "UTC"

# 3. 目标频率
TARGET_FREQ = '15min'

# 4. 变量映射字典
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
    """计算理论晴空辐射，用于物理插值"""
    # 注意：这里的 times 必须带有正确的时区信息 (UTC)
    location = pvlib.location.Location(lat, lon, altitude=alt)
    cs = location.get_clearsky(times, model='ineichen')
    return cs

def physics_aware_upsample(df_1h, target_freq='15min'):
    """
    [物理升采样] 1h NWP -> 15min
    """
    print(f"   [物理处理] 正在执行 NWP 升采样 (1h -> {target_freq})...")
    
    df_1h = df_1h.sort_index()
    
    # 生成目标 15min 时间轴 (继承 UTC 时区)
    target_index = pd.date_range(
        start=df_1h.index[0], 
        end=df_1h.index[-1], 
        freq=target_freq, 
        tz=df_1h.index.tz 
    )
    
    df_result = pd.DataFrame(index=target_index)
    
    # 计算 ClearSky (此时 index 已经是 UTC，计算会准确)
    cs_1h = get_clearsky_profile(df_1h.index, LATITUDE, LONGITUDE, ALTITUDE)
    cs_target = get_clearsky_profile(target_index, LATITUDE, LONGITUDE, ALTITUDE)
    
    # --- A. 辐射：晴空指数插值 ---
    rad_params = {
        'shortwave_radiation (W/m²)': ('ghi', 'NWP_GHI'),
        'direct_normal_irradiance (W/m²)': ('dni', 'NWP_DNI'),
        'diffuse_radiation (W/m²)': ('dhi', 'NWP_DHI')
    }
    
    for raw_col, (cs_type, new_name) in rad_params.items():
        if raw_col in df_1h.columns:
            # 1. 计算 hourly Kc
            k_series_1h = df_1h[raw_col] / (cs_1h[cs_type] + 1e-6)
            k_series_1h = k_series_1h.fillna(0)
            k_series_1h[cs_1h[cs_type] < 5.0] = 0 
            
            # 2. 线性插值 Kc 到 15min
            k_series_target = k_series_1h.reindex(target_index).interpolate(method='linear').fillna(0)
            
            # 3. 还原
            df_result[new_name] = k_series_target * cs_target[cs_type]
            df_result[new_name] = df_result[new_name].clip(lower=0)

    # --- B. 气象：Cubic 插值 ---
    meteo_cols = ['temperature_2m (°C)', 'wind_speed_10m (km/h)', 'relative_humidity_2m (%)']
    for col in meteo_cols:
        if col in df_1h.columns:
            new_name = NWP_COL_MAP.get(col, col)
            try:
                df_result[new_name] = df_1h[col].reindex(target_index).interpolate(method='cubic').fillna(method='ffill').fillna(method='bfill')
            except:
                df_result[new_name] = df_1h[col].reindex(target_index).interpolate(method='linear')

    # --- C. 其他：Linear 插值 ---
    linear_cols = ['cloud_cover (%)', 'precipitation (mm)']
    for col in linear_cols:
        if col in df_1h.columns:
            new_name = NWP_COL_MAP.get(col, col)
            df_result[new_name] = df_1h[col].reindex(target_index).interpolate(method='linear').fillna(0)
            
    return df_result

def prepare_data():
    base_dir = Path("outputs/clean")
    pv_path = base_dir / "Final_Dataset_With_Features_5min_UTC.csv"
    nwp_path = base_dir / "NWP.csv"
    output_path = base_dir / "dataset_ready_for_research_15min.csv"

    print(f"--- [Step 1] 数据准备: 15min 日前预测数据集 (UTC) ---")
    print(f"--- 实测源: {pv_path}")
    print(f"--- 目标频率: {TARGET_FREQ}")

    # ================= 1. 处理 PV 数据 (5min UTC -> 15min UTC) =================
    if not pv_path.exists():
        print(f"❌ 错误: 找不到 {pv_path}")
        return
    
    print("1. 处理实测数据 (保持 UTC)...")
    df_pv = pd.read_csv(pv_path)
    
    time_col = df_pv.columns[0]
    if 'Timestamp_UTC' in df_pv.columns: time_col = 'Timestamp_UTC'
    
    # 解析时间，直接声明为 UTC (因为你的文件名说了它是 UTC)
    df_pv[time_col] = pd.to_datetime(df_pv[time_col], utc=True)
    df_pv.set_index(time_col, inplace=True)
    
    # 降采样
    df_pv_15min = df_pv.resample(TARGET_FREQ, closed='left', label='left').mean()
    
    rename_map = {
        'Power_Actual': 'Target_Power',
        'Power_Simulated': 'Sim_Power',
    }
    df_pv_15min = df_pv_15min.rename(columns=rename_map)

    # ================= 2. 处理 NWP 数据 (LA Time -> UTC -> 15min) =================
    if not nwp_path.exists():
        print(f"❌ 错误: 找不到 {nwp_path}")
        return
    print("2. 处理 NWP 数据 (LA Time -> UTC)...")
    
    try:
        df_nwp = pd.read_csv(nwp_path, skiprows=3)
        if 'time' not in df_nwp.columns: df_nwp = pd.read_csv(nwp_path)
    except:
        df_nwp = pd.read_csv(nwp_path)

    # 2.1 [核心修改] 正确处理洛杉矶时区
    # 第一步：解析时间
    df_nwp['time'] = pd.to_datetime(df_nwp['time'])
    
    # 第二步：声明原始数据是洛杉矶时间 (处理夏令时 DST)
    # ambiguous='NaT' 会处理夏令时切换时的模糊时间
    df_nwp['time'] = df_nwp['time'].dt.tz_localize('America/Los_Angeles', ambiguous='NaT', nonexistent='shift_forward')
    
    # 第三步：统一转换到 UTC
    df_nwp['time'] = df_nwp['time'].dt.tz_convert('UTC')
    
    df_nwp.set_index('time', inplace=True)
    # 删除因夏令时转换可能产生的空值或重复
    df_nwp = df_nwp[df_nwp.index.notnull()]
    df_nwp = df_nwp[~df_nwp.index.duplicated()]

    # 2.2 物理升采样 (在 UTC 坐标系下进行)
    df_nwp_15min = physics_aware_upsample(df_nwp, target_freq=TARGET_FREQ)

    # ================= 3. 数据合并 =================
    print("3. 合并 PV 与 NWP...")
    
    # 取交集 (确保 UTC 时间对齐)
    common_index = df_pv_15min.index.intersection(df_nwp_15min.index)
    
    if len(common_index) == 0:
        print("❌ 错误: PV 和 NWP 时间无重叠！")
        print(f"   PV (UTC): {df_pv_15min.index[0]} ~ {df_pv_15min.index[-1]}")
        print(f"   NWP (UTC): {df_nwp_15min.index[0]} ~ {df_nwp_15min.index[-1]}")
        return

    final_df = pd.concat([df_pv_15min.loc[common_index], df_nwp_15min.loc[common_index]], axis=1)
    final_df = final_df.fillna(method='ffill').fillna(0)

    # ================= 4. 夜间物理清洗 =================
    print("4. 清洗夜间数据...")
    # 注意：计算太阳位置时，location 初始化为 UTC 即可，经纬度决定了物理位置
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz='UTC')
    solpos = loc.get_solarposition(final_df.index)
    is_night = solpos['elevation'] < -5.0
    
    rad_cols = ['NWP_GHI', 'NWP_DNI', 'NWP_DHI', 'Target_Power', 'Sim_Power']
    for col in rad_cols:
        if col in final_df.columns:
            final_df.loc[is_night, col] = 0.0

    # ================= 5. 保存 =================
    final_df.to_csv(output_path)
    print(f"\n[✅ 成功] 15min 数据集已保存至: {output_path}")
    print(f"   时区: UTC")
    print(f"   时间范围: {final_df.index[0]} ~ {final_df.index[-1]}")
    print(f"   数据形状: {final_df.shape}")

if __name__ == "__main__":
    prepare_data()