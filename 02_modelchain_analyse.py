# -*- coding: utf-8 -*-
# 版本V5.6：集成物理模型链 + 纯净过滤 + Origin绘图数据导出

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib
from pathlib import Path
from scipy.optimize import curve_fit
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# ===================================================================
# 【重要】参数配置
# ===================================================================
# 1. 地理位置与时区
LATITUDE = 34.05
LONGITUDE = -118.24
ALTITUDE = 71
TIMEZONE = "America/Los_Angeles"

# 2. 文件路径
INP = Path("outputs/clean")
FILE_NAME = "merged_pv_and_weather_data.csv" 

OUT = Path("outputs/pv_phys_baseline_2d"); OUT.mkdir(parents=True, exist_ok=True)
PARAM_OUT = INP / "physics_params.csv" # 参数保存路径

# 【新增】Origin 绘图数据保存路径
SCATTER_OUT = OUT / "origin_scatter_data.csv"
FIT_LINE_OUT = OUT / "origin_fit_line_data.csv"

P_RATED_KW = 30.1

# 3. 分析参数
TIME_RESOLUTION = '15min' 
LAG_TO_CORRECT_MINUTES = 0

# 4. 物理系统参数
SURFACE_TILT = 22.5       
SURFACE_AZIMUTH = 195   
# ===================================================================

# --- 1. 数据加载与预处理 ---
print("--- 步骤1：加载原始数据 (含气象信息) ---")
def read_csv_tz(path_csv: Path, tz: str) -> pd.DataFrame:
    # 兼容两种可能的 CSV 格式
    df = pd.read_csv(path_csv)
    
    # 优先查找 'Target_Power' (新数据集) 或 'Power_Actual' (旧数据集)
    pcol = next((c for c in df.columns if c in ['Target_Power', 'Power_Actual', 'p_kw']), None)
    
    # 查找时间列
    time_col = next((c for c in df.columns if 'time' in c.lower() or 'ts' in c.lower()), df.columns[0])
    
    # 解析时间
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(tz)
    df.set_index(time_col, inplace=True)
    return df, pcol

full_path = INP / FILE_NAME
if not full_path.exists():
    print(f"Warning: {FILE_NAME} 不存在，尝试读取 dataset_ready_for_research.csv")
    full_path = INP / "dataset_ready_for_research.csv"

df_raw, p_col_name = read_csv_tz(full_path, TIMEZONE)

P_actual_kw = pd.to_numeric(df_raw[p_col_name], errors="coerce").fillna(0.0).clip(lower=0.0)

# 兼容气象数据列名 (支持新旧两种格式)
if 'Obs_Temp' in df_raw.columns:
    temp_air = df_raw['Obs_Temp']
elif 'temp_air' in df_raw.columns:
    temp_air = df_raw['temp_air']
else:
    print("注意: 未找到气温列，使用 20°C 默认值")
    temp_air = pd.Series(20, index=df_raw.index)

# 兼容风速
wind_speed = pd.Series(1, index=df_raw.index) # 默认
if 'NWP_Wind' in df_raw.columns: wind_speed = df_raw['NWP_Wind']
elif 'wind_speed' in df_raw.columns: wind_speed = df_raw['wind_speed']

print("步骤1：数据加载完成。")

# --- 1b. 时间戳修正 ---
if LAG_TO_CORRECT_MINUTES != 0:
    print(f"--- 步骤1b：时间修正 {LAG_TO_CORRECT_MINUTES} min ---")
    shift = pd.to_timedelta(LAG_TO_CORRECT_MINUTES, unit='m')
    P_actual_kw.index = P_actual_kw.index - shift
    temp_air.index = temp_air.index - shift
    wind_speed.index = wind_speed.index - shift

# --- 2. 构建物理模型链 ---
print("--- 步骤2：构建物理模型链 (Physical Model Chain) ---")
location_obj = pvlib.location.Location(latitude=LATITUDE, longitude=LONGITUDE, tz=TIMEZONE, altitude=ALTITUDE)
temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

system = PVSystem(
    surface_tilt=SURFACE_TILT,
    surface_azimuth=SURFACE_AZIMUTH,
    module_parameters={'pdc0': P_RATED_KW * 1000, 'gamma_pdc': -0.005}, 
    inverter_parameters={'pdc0': P_RATED_KW * 1000, 'eta_inv_nom': 0.95},
    temperature_model_parameters=temp_params
)

mc = ModelChain(system, location_obj, aoi_model='physical', spectral_model='no_loss')

print("   正在计算晴空辐射...")
cs = location_obj.get_clearsky(df_raw.index)

# 如果输入数据里有 GHI (Obs_GHI 或 ghi)，最好用 DISC 模型估算 DNI
# 这里简化处理，直接用 ClearSky 的 DNI/DHI 分量
# 但必须用实测温度修正效率
weather_clearsky = pd.DataFrame({
    'ghi': cs['ghi'], 'dni': cs['dni'], 'dhi': cs['dhi'],
    'temp_air': temp_air, 'wind_speed': wind_speed
})

print("   正在运行物理仿真...")
mc.run_model(weather_clearsky)
P_phys_clearsky = mc.results.ac.fillna(0).clip(lower=0) / 1000.0
P_clearsky_kw = P_phys_clearsky
print("步骤2：物理晴空基准 (P_phys) 计算完成。")

# --- 3. 计算“云瞬变” ---
cloud_transient = (P_actual_kw - P_clearsky_kw)
cloud_transient[P_clearsky_kw < 0.001 * P_RATED_KW] = 0
print("步骤3：已计算波动分量。")

# --- 4. 计算波动强度 ---
print(f"--- 步骤4：以 {TIME_RESOLUTION} 窗口计算统计指标 ---")

df_calc = pd.DataFrame({'P': P_actual_kw, 'P_clear': P_clearsky_kw})
sigma_pv = df_calc['P'].resample(TIME_RESOLUTION).std().rename('sigma_PV')
Pst_mean = df_calc['P_clear'].resample(TIME_RESOLUTION).mean().rename('Px_clearsky')

df_analysis = pd.concat([Pst_mean, sigma_pv], axis=1).dropna()
df_analysis = df_analysis[df_analysis['Px_clearsky'] > 0.001 * P_RATED_KW]
df_analysis['I_F_PV'] = df_analysis['sigma_PV'] / df_analysis['Px_clearsky']
df_analysis = df_analysis.rename(columns={'sigma_PV': 'sigma_cloud'})

# ==============================================================================
# --- 步骤 5：数据过滤 (只去除晴天底噪，保留原始散点) ---
# ==============================================================================
print("--- 步骤5：执行数据清洗 (不分箱) ---")

# 1. 提取原始数据
x_raw = df_analysis['Px_clearsky'] / P_RATED_KW
y_raw = df_analysis['I_F_PV']

# 2. 核心过滤：去除 "晴空底噪"
# 逻辑：只保留 I_F > 0.02 (2%) 的"活跃波动点"
mask_active = (x_raw > 0.008) & (y_raw > 0.02) & np.isfinite(x_raw) & np.isfinite(y_raw)
# mask_active =(y_raw > 0.01) & np.isfinite(x_raw) & np.isfinite(y_raw)
x_active = x_raw[mask_active]
y_active = y_raw[mask_active]

print(f"原始数据点: {len(x_raw)} -> 活跃波动点(拟合用): {len(x_active)}")

# ==============================================================================
# --- 步骤 6：基于活跃散点的直接拟合与可视化 ---
# ==============================================================================
def power_law_model(p, a, beta, c):
    return a * np.power(p, beta) + c

try:
    print("--- 步骤6：执行原始散点拟合 ---")
    
    # 直接对成千上万个活跃点进行拟合
    popt, _ = curve_fit(power_law_model, x_active, y_active, 
                        p0=[0.1, -0.5, 0.02], 
                        bounds=([0, -3, 0], [10, 0, 1]),
                        maxfev=5000)
    a, beta, c = popt
    
    print("="*40)
    print(f"【拟合结果】(基于筛选后的原始散点):")
    print(f"I_F = {a:.4f} * P_phys^{beta:.4f} + {c:.4f}")
    print("="*40)

    # --- 保存拟合参数 (供 Transformer 使用) ---
    params_df = pd.DataFrame({
        'parameter': ['a', 'beta', 'c'],
        'value': [a, beta, c]
    })
    params_df.to_csv(PARAM_OUT, index=False)
    print(f"已将物理参数保存至: {PARAM_OUT}")
    
    # --- 【核心新增】保存 Origin 绘图数据 ---
    # 1. 散点数据 (X: P_phys, Y: I_F)
    scatter_df = pd.DataFrame({
        'P_phys_pu': x_active,
        'I_F': y_active
    })
    scatter_df.to_csv(SCATTER_OUT, index=False)
    print(f"Origin散点数据已保存: {SCATTER_OUT}")
    
    # 2. 拟合线数据 (X: P_phys_line, Y: I_F_fitted)
    x_line = np.linspace(0.002, 1.0, 100)
    y_line = power_law_model(x_line, *popt)
    fit_df = pd.DataFrame({
        'P_phys_pu': x_line,
        'I_F_fitted': y_line
    })
    fit_df.to_csv(FIT_LINE_OUT, index=False)
    print(f"Origin拟合线数据已保存: {FIT_LINE_OUT}")

    # --- 绘图 ---
    plt.figure(figsize=(10, 7))
    plt.scatter(x_active, y_active, alpha=0.15, s=5, color='dodgerblue', label='Active Volatility ($I_F > 0.02$)')
    
    # 绘制拟合曲线
    plt.plot(x_line, y_line, 'r-', linewidth=3, label=f'Fit: $\\beta$={beta:.3f}')
    
    plt.title(f'Fluctuation Intensity Analysis (Filtered Only, No Binning)\nResolution: {TIME_RESOLUTION}', fontsize=14)
    plt.xlabel('Physical Baseline Power $P_{phys}$ (p.u.)', fontsize=12)
    plt.ylabel('Fluctuation Intensity $I_F$', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 1.5) # 聚焦有效范围
    
    out_png = OUT / "phys_baseline_filtered_fit.png"
    plt.savefig(out_png, dpi=150)
    # plt.show() # 如果在服务器运行，注释掉这一行
    print(f"图表已保存至: {out_png}")

except Exception as e:
    print(f"拟合失败: {e}")

# --- 7. 保存分析数据 ---
out_csv = OUT / "phys_baseline_analysis_data.csv"
df_analysis.to_csv(out_csv)
print(f"分析数据已保存至: {out_csv}")