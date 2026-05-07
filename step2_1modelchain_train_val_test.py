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
# 【配置区域】
# ===================================================================
# 1. 地理位置与时区
LATITUDE = 37.42
LONGITUDE = -122.15
ALTITUDE = 71
TIMEZONE = "UTC"

# 2. 路径配置 (使用绝对路径防错)
CURRENT_DIR = Path(__file__).parent.absolute()
INP = CURRENT_DIR / "rawdata"
FILE_NAME = INP / "merged_2018_2019_5min_UTC.csv"

# 输出路径
OUT = Path("outputs/Scatter")
OUT.mkdir(parents=True, exist_ok=True)
PARAM_OUT = INP / "physics_params.csv"
SCATTER_OUT = OUT / "origin_scatter_data.csv"      # 原始散点
BINNED_OUT = OUT / "origin_binned_data.csv"        # 分箱后的规律点 (Origin用)
FIT_LINE_OUT = OUT / "origin_fit_line_data.csv"    # 拟合曲线 (Origin用)

print(f"\n[调试信息]")
print(f"读取文件: {FILE_NAME}")
print(f"存在状态: {FILE_NAME.exists()}")
print(f"="*30 + "\n")

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
    pcol = next((c for c in df.columns if c in ['Target_Power', 'power_kw', 'p_kw']), None)
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
    module_parameters={'pdc0': P_RATED_KW * 1000, 'gamma_pdc': -0.003},
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
# --- [关键修改] 步骤 5：70/15/15 严格切分与防泄露处理 ---
# ==============================================================================
print("--- 步骤5：执行严格的数据切分 (70% Train / 15% Val / 15% Test) ---")

# 1. 计算切分点索引
n_total = len(df_analysis)
idx_train_end = int(n_total * 0.70)
idx_val_end = int(n_total * (0.70 + 0.15))

# 2. 按时间序列切分
df_train = df_analysis.iloc[:idx_train_end].copy()
df_val   = df_analysis.iloc[idx_train_end:idx_val_end].copy()
df_test  = df_analysis.iloc[idx_val_end:].copy()

print(f"总样本数: {n_total}")
print(f"  - 训练集 (70%): {len(df_train)} [拟合参数来源]")
print(f"  - 验证集 (15%): {len(df_val)}   [模型调优]")
print(f"  - 测试集 (15%): {len(df_test)}  [最终评估]")

# 3. 定义过滤函数 (复用逻辑)
def filter_active_points(df):
    x_raw = df['Px_clearsky'] / P_RATED_KW
    y_raw = df['I_F_PV']
    # 过滤逻辑：P > 0.008 且 波动 > 0.02 (只提取活跃波动用于拟合/验证)
    mask = (x_raw > 0.008) & (y_raw > 0.02) & np.isfinite(x_raw) & np.isfinite(y_raw)
    return x_raw[mask], y_raw[mask]

# 4. 分别提取数据
x_train, y_train = filter_active_points(df_train)
x_val, y_val     = filter_active_points(df_val)
x_test, y_test   = filter_active_points(df_test)

print(f"训练集活跃点: {len(x_train)}")
print(f"验证集活跃点: {len(x_val)}")
print(f"测试集活跃点: {len(x_test)}")

# ==============================================================================
# --- 步骤 6：基于训练集的拟合与全流程验证 ---
# ==============================================================================
def power_law_model(p, a, beta, c):
    return a * np.power(p, beta) + c

try:
    print("--- 步骤6：执行拟合 (仅基于训练集 70%) ---")
    
    # [核心] 只使用 x_train, y_train 进行拟合
    popt, pcov = curve_fit(power_law_model, x_train, y_train,
                           p0=[0.1, -0.5, 0.02],
                           bounds=([0, -3, 0], [10, 0, 1]),
                           maxfev=5000)
    a, beta, c = popt
    
    print("="*40)
    print(f"【拟合结果】(仅基于训练集):")
    print(f"I_F = {a:.4f} * P_phys^{beta:.4f} + {c:.4f}")
    print("="*40)

    # --- 保存参数 (供后续神经网络 Loss 使用) ---
    params_df = pd.DataFrame({'parameter': ['a', 'beta', 'c'], 'value': [a, beta, c]})
    params_df.to_csv(PARAM_OUT, index=False)

    # --- [创新点绘图] 物理规律平稳性验证 (Train vs Val vs Test) ---
    plt.figure(figsize=(10, 7))
    
    # 1. 绘制训练集背景 (灰色)
    plt.scatter(x_train, y_train, alpha=0.05, s=2, color='gray', label='Train (70%)')
    
    # 2. 绘制验证集 (绿色) - 可选，展示中间态
    plt.scatter(x_val, y_val, alpha=0.1, s=2, color='limegreen', label='Val (15%)')

    # 3. 绘制测试集 (蓝色) - 最重要
    plt.scatter(x_test, y_test, alpha=0.1, s=2, color='dodgerblue', label='Test (15%)')
    
    # 4. 绘制拟合曲线 (红色, 基于训练集)
    # x_line = np.linspace(0.01, 1.0, 100)
    # y_line = power_law_model(x_line, *popt)
    # plt.plot(x_line, y_line, 'r-', linewidth=3, label='Physics-Guided Law (from Train)')
    
    plt.title('Stationarity Validation across Splits (70/15/15)', fontsize=14)
    plt.xlabel('Physical Baseline Power $P_{phys}$ (p.u.)', fontsize=12)
    plt.ylabel('Fluctuation Intensity $I_F$', fontsize=12)
    plt.legend(loc='upper right', markerscale=3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 2.5)
    
    out_png = OUT / "stationarity_check_3split.png"
    plt.savefig(out_png, dpi=150)
    print(f"验证图表已保存: {out_png}")
    
    # --- 保存 Origin 数据 (三份独立保存，方便画图) ---
    print("正在导出 Origin 绘图数据...")
    pd.DataFrame({'P_train': x_train, 'I_F_train': y_train}).to_csv(OUT/"origin_train_70.csv", index=False)
    pd.DataFrame({'P_val': x_val,     'I_F_val': y_val}).to_csv(OUT/"origin_val_15.csv", index=False)
    pd.DataFrame({'P_test': x_test,   'I_F_test': y_test}).to_csv(OUT/"origin_test_15.csv", index=False)
    # pd.DataFrame({'P_fit': x_line,    'I_F_fit': y_line}).to_csv(FIT_LINE_OUT, index=False)
    print("Origin 数据导出完成。")

except Exception as e:
    print(f"拟合失败: {e}")
    import traceback
    traceback.print_exc()
# --- 7. 保存分析数据 ---
out_csv = OUT / "phys_baseline_analysis_data.csv"
df_analysis.to_csv(out_csv)
print(f"分析数据已保存至: {out_csv}")