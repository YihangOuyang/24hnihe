# -*- coding: utf-8 -*-
"""
Generate_1min_Dataset_NoLeakage.py

功能：
1. 生成 1分钟 分辨率的物理增强数据集。
2. 严格防止未来信息泄露 (使用 ffill)。
3. 输出文件：Final_Dataset_1min_Physics_UTC.csv
"""

import os
import pandas as pd
import numpy as np
import pvlib
import matplotlib.pyplot as plt

# ================= 1. 全局配置 =================
BASE_DIR = r'../data'
NSRDB_FILE = os.path.join(BASE_DIR, 'stanford_dataset.csv')
ACTUAL_FILE = os.path.join(BASE_DIR, '2019_pv_daytime_only.csv')
# [修改] 仅输出 1分钟 文件
OUTPUT_FILE_1MIN = os.path.join(BASE_DIR, 'Final_Dataset_1min_Physics_UTC.csv')

# 系统参数 (保持一致)
SYSTEM_CONFIG = {
    'tilt': 37.43,
    'azimuth': 180,
    'capacity_dc': 30000,
    'capacity_ac': 25000,
    'albedo': 0.25,
    'gamma_pdc': -0.45/100,
    'temp_model': {'a': -3.56, 'b': -0.075, 'deltaT': 3}
}

def main():
    print("--- 🚀 开始处理：生成 1分钟 无泄露数据集 ---")

    # ================= 2. 读取并预处理 NSRDB 数据 =================
    print(f"\n[1/5] 读取 NSRDB 数据...")
    try:
        with open(NSRDB_FILE, 'r') as f:
            df_weather, metadata = pvlib.iotools.psm3.parse_psm3(f, map_variables=True)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {NSRDB_FILE}")
        return

    # [时区修正] 
    # 假设源数据是 UTC-8 (Etc/GMT+8)
    df_weather.index = pd.to_datetime(df_weather.index)
    if df_weather.index.tz is None:
        print("      提示：应用 'Etc/GMT+8' 时区 (NSRDB)")
        df_weather.index = df_weather.index.tz_localize('Etc/GMT+8')
    
    # 统一转为 UTC
    df_weather.index = df_weather.index.tz_convert('UTC')

    # 锁定原始 10分钟 数据点
    df_10min = df_weather[df_weather.index.minute % 10 == 0].copy()
    print(f"      原始天气数据加载完毕 (10min频率)")


    # ================= 3. [防泄露核心] 物理插值 =================
    print(f"\n[2/5] 执行因果物理插值 (Forward Fill CSI)...")

    # 3.1 创建 1分钟 时间轴 (UTC)
    times_1min = pd.date_range(start=df_10min.index[0], end=df_10min.index[-1], freq='1T', tz='UTC')
    location = pvlib.location.Location(metadata['latitude'], metadata['longitude'])

    # 3.2 计算晴空模型
    print("      计算晴空模型...")
    cs_10min = location.get_clearsky(df_10min.index, model='ineichen')
    cs_1min = location.get_clearsky(times_1min, model='ineichen')

    # 3.3 计算晴空指数 (CSI)
    def calculate_csi_safe(actual, clearsky, threshold=5):
        csi = actual / clearsky
        mask = clearsky < threshold
        csi[mask] = 0
        csi = csi.clip(0, 2.5) 
        return csi

    ghi_csi = calculate_csi_safe(df_10min['ghi'], cs_10min['ghi'])
    dni_csi = calculate_csi_safe(df_10min['dni'], cs_10min['dni'])
    dhi_csi = calculate_csi_safe(df_10min['dhi'], cs_10min['dhi'])

    # 3.4 [防泄露] 使用 Forward Fill (ffill)
    print("      ⚠️ 正在应用 Forward Fill (防泄露插值)...")
    
    ghi_csi_1min = ghi_csi.reindex(times_1min).ffill().fillna(0)
    dni_csi_1min = dni_csi.reindex(times_1min).ffill().fillna(0)
    dhi_csi_1min = dhi_csi.reindex(times_1min).ffill().fillna(0)

    # 3.5 还原辐照度
    df_1min = pd.DataFrame(index=times_1min)
    df_1min['ghi'] = ghi_csi_1min * cs_1min['ghi']
    df_1min['dni'] = dni_csi_1min * cs_1min['dni']
    df_1min['dhi'] = dhi_csi_1min * cs_1min['dhi']

    # 3.6 其他气象数据的填充 (ffill)
    other_cols = ['temp_air', 'wind_speed']
    for col in other_cols:
        if col in df_10min.columns:
            df_1min[col] = df_10min[col].reindex(times_1min).ffill()
    
    df_1min[df_1min < 0] = 0


    # ================= 4. 运行 PVWatts (1min) =================
    print(f"\n[3/5] 运行光伏物理模型 (1min)...")
    
    solpos_1min = location.get_solarposition(times_1min, temperature=df_1min['temp_air'])
    
    dni_extra = pvlib.irradiance.get_extra_radiation(times_1min, solar_constant=1361.1, method='nrel')
    airmass_rel = pvlib.atmosphere.get_relative_airmass(solpos_1min['apparent_zenith'], model='kastenyoung1989')

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=SYSTEM_CONFIG['tilt'],
        surface_azimuth=SYSTEM_CONFIG['azimuth'],
        solar_zenith=solpos_1min['apparent_zenith'],
        solar_azimuth=solpos_1min['azimuth'],
        dni=df_1min['dni'],
        ghi=df_1min['ghi'],
        dhi=df_1min['dhi'],
        dni_extra=dni_extra,
        airmass=airmass_rel,
        albedo=SYSTEM_CONFIG['albedo'],
        model='perez'
    )

    cell_temp = pvlib.temperature.sapm_cell(
        poa_global=poa['poa_global'],
        temp_air=df_1min['temp_air'],
        wind_speed=df_1min['wind_speed'],
        a=SYSTEM_CONFIG['temp_model']['a'], 
        b=SYSTEM_CONFIG['temp_model']['b'], 
        deltaT=SYSTEM_CONFIG['temp_model']['deltaT'], 
        irrad_ref=1000.0
    )

    pv_dc = pvlib.pvsystem.pvwatts_dc(
        g_poa_effective=poa['poa_global'],
        temp_cell=cell_temp,
        pdc0=SYSTEM_CONFIG['capacity_dc'],
        gamma_pdc=SYSTEM_CONFIG['gamma_pdc'],
        temp_ref=25.0
    )

    loss_fraction = pvlib.pvsystem.pvwatts_losses(
        soiling=2, shading=0, snow=3, mismatch=0, wiring=2, 
        connections=0, lid=0, nameplate_rating=0, age=0, availability=3
    ) / 100

    pdc_input = pv_dc * (1 - loss_fraction)
    
    pv_ac = pvlib.inverter.pvwatts(
        pdc=pdc_input, 
        pdc0=SYSTEM_CONFIG['capacity_ac'] / 0.975, 
        eta_inv_nom=0.975,
        eta_inv_ref=0.9637
    )

    df_1min['Power_Simulated'] = pv_ac.fillna(0) / 1000 # kW
    df_1min['Power_Simulated'] = df_1min['Power_Simulated'].clip(lower=0)


    # ================= 5. 加载并合并实测数据 =================
    print(f"\n[4/5] 合并实测数据 (对齐到 UTC)...")
    if os.path.exists(ACTUAL_FILE):
        df_actual = pd.read_csv(ACTUAL_FILE)
        df_actual.iloc[:,0] = pd.to_datetime(df_actual.iloc[:,0])
        df_actual.set_index(df_actual.columns[0], inplace=True)
        df_actual.rename(columns={df_actual.columns[0]: 'Power_Actual'}, inplace=True)
        
        # [时区修正] US/Pacific -> UTC
        if df_actual.index.tz is not None: df_actual = df_actual.tz_localize(None)
        
        # 声明原始时区为 Pacific
        df_actual = df_actual.tz_localize('US/Pacific', ambiguous='NaT', nonexistent='shift_forward')
        # 转换到 UTC 以匹配模拟数据
        df_actual_utc = df_actual.tz_convert('UTC')
        
        # 合并
        df_merged = df_1min.join(df_actual_utc, how='inner')
    else:
        df_merged = df_1min
        print("⚠️ 警告：无实测数据")

    # 计算误差列 (可选)
    if 'Power_Actual' in df_merged.columns:
        df_merged['Error'] = df_merged['Power_Simulated'] - df_merged['Power_Actual']

    # ================= 6. 保存 1分钟 数据 =================
    print(f"\n[5/5] 保存 1分钟 文件: {OUTPUT_FILE_1MIN}")
    df_merged.to_csv(OUTPUT_FILE_1MIN, index_label='Timestamp_UTC')
    print(f"✅ 处理完成！")
    print(f"   数据行数: {len(df_merged)}")
    print(f"   包含列: {df_merged.columns.tolist()}")
    
    # 验证绘图
    plt.figure(figsize=(12, 6))
    # 随便取中间的一天来画图验证
    mid_idx = len(df_merged) // 2
    subset = df_merged.iloc[mid_idx : mid_idx + 1440] # 1440分钟 = 1天
    
    plt.plot(subset.index, subset['Power_Simulated'], label='Simulated (1min No Leakage)', color='blue', alpha=0.8)
    if 'Power_Actual' in subset.columns:
        plt.plot(subset.index, subset['Power_Actual'], label='Actual (1min)', color='orange', alpha=0.6)
    
    plt.title("Verification: 1-min Resolution (No Future Leakage)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == '__main__':
    main()