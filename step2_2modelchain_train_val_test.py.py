# -*- coding: utf-8 -*-
# 版本V6.0：集成物理模型链 + Interleaved Split + 平稳性验证
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib
from pathlib import Path
from scipy.optimize import curve_fit
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import warnings

warnings.filterwarnings("ignore")

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
FILE_NAME = INP / "merged_2018_2019_5min_UTC_cleaned.csv"

# 输出路径
OUT = CURRENT_DIR / "outputs" / "Scatter"
OUT.mkdir(parents=True, exist_ok=True)
PARAM_OUT = INP / "physics_params.csv" # 输出给Step3/4使用

# 3. 分析参数
P_RATED_KW = 30.1
TIME_RESOLUTION = '15min'
LAG_TO_CORRECT_MINUTES = 0

# 4. 物理系统参数
SURFACE_TILT = 22.5      
SURFACE_AZIMUTH = 195  

print(f"\n[调试信息]")
print(f"读取文件: {FILE_NAME}")
print(f"存在状态: {FILE_NAME.exists()}")
print(f"="*30 + "\n")

# ===================================================================
# --- 1. 数据加载与预处理 ---
print("--- 步骤1：加载原始数据 (含气象信息) ---")
def read_csv_tz(path_csv: Path, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    pcol = next((c for c in df.columns if c in ['Target_Power', 'power_kw', 'p_kw']), None)
    time_col = next((c for c in df.columns if 'time' in c.lower() or 'ts' in c.lower()), df.columns[0])
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(tz)
    df.set_index(time_col, inplace=True)
    return df, pcol

full_path = INP / FILE_NAME
if not full_path.exists():
    print(f"Warning: {FILE_NAME} 不存在，尝试读取 dataset_ready_for_research.csv")
    full_path = INP / "dataset_ready_for_research.csv"
    
df_raw, p_col_name = read_csv_tz(full_path, TIMEZONE)
P_actual_kw = pd.to_numeric(df_raw[p_col_name], errors="coerce").fillna(0.0).clip(lower=0.0)

# 兼容气象数据列名
if 'Obs_Temp' in df_raw.columns: temp_air = df_raw['Obs_Temp']
elif 'temp_air' in df_raw.columns: temp_air = df_raw['temp_air']
else: temp_air = pd.Series(20, index=df_raw.index)

if 'NWP_Wind' in df_raw.columns: wind_speed = df_raw['NWP_Wind']
elif 'wind_speed' in df_raw.columns: wind_speed = df_raw['wind_speed']
else: wind_speed = pd.Series(1, index=df_raw.index)

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
# (此步骤仅为逻辑完整性保留，实际拟合不依赖此中间变量)
cloud_transient = (P_actual_kw - P_clearsky_kw)
cloud_transient[P_clearsky_kw < 0.001 * P_RATED_KW] = 0

# --- 4. 计算波动强度 ---
print(f"--- 步骤4：以 {TIME_RESOLUTION} 窗口计算统计指标 ---")

df_calc = pd.DataFrame({'P': P_actual_kw, 'P_clear': P_clearsky_kw})
sigma_pv = df_calc['P'].resample(TIME_RESOLUTION).std().rename('sigma_PV')
Pst_mean = df_calc['P_clear'].resample(TIME_RESOLUTION).mean().rename('Px_clearsky')

df_analysis = pd.concat([Pst_mean, sigma_pv], axis=1).dropna()
# 过滤夜间微小数值
df_analysis = df_analysis[df_analysis['Px_clearsky'] > 0.001 * P_RATED_KW]
# 计算归一化波动强度 I_F
df_analysis['I_F_PV'] = df_analysis['sigma_PV'] / df_analysis['Px_clearsky']

# ==============================================================================
# --- [关键修改] 步骤 5：Interleaved Split 切分 (与 Step 3 保持一致) ---
# ==============================================================================
print("--- 步骤5：执行交替切分 (Interleaved Split: 4:1:1) ---")

start_date = df_analysis.index[0]
days_diff = (df_analysis.index - start_date).days
biweek_idx = days_diff // 14
cycle_idx = biweek_idx % 6

# 4个双周 Train, 1个 Val, 1个 Test
train_mask = np.isin(cycle_idx, [0, 1, 2, 3])
val_mask   = (cycle_idx == 4)
test_mask  = (cycle_idx == 5)

df_train = df_analysis[train_mask].copy()
df_val   = df_analysis[val_mask].copy()
df_test  = df_analysis[test_mask].copy()

print(f"总样本数: {len(df_analysis)}")
print(f"  - 训练集: {len(df_train)} (67%)")
print(f"  - 验证集: {len(df_val)}   (16.5%)")
print(f"  - 测试集: {len(df_test)}  (16.5%)")

# 3. 定义过滤函数 (复用逻辑)
def filter_active_points(df):
    # X轴: 归一化物理功率 (p.u.)
    x_raw = df['Px_clearsky'] / P_RATED_KW
    # Y轴: 归一化波动强度
    y_raw = df['I_F_PV']
    
    # 过滤逻辑：P > 0.008 且 波动 > 0.02 (只提取活跃波动用于拟合/验证)
    mask = (x_raw > 0.008) & (y_raw > 0.02) & np.isfinite(x_raw) & np.isfinite(y_raw)
    return x_raw[mask], y_raw[mask]

# 4. 分别提取数据
x_train, y_train = filter_active_points(df_train)
x_val, y_val     = filter_active_points(df_val)
x_test, y_test   = filter_active_points(df_test)

print(f"活跃点数: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}")

# ==============================================================================
# --- 步骤 6：基于训练集的拟合与全流程验证 ---
# ==============================================================================
def power_law_model(p, a, beta, c):
    return a * np.power(p, beta) + c

try:
    print("--- 步骤6：执行拟合 (仅基于训练集) ---")
    
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

    # --- 保存参数 ---
    params_df = pd.DataFrame({'parameter': ['a', 'beta', 'c'], 'value': [a, beta, c]})
    params_df.to_csv(PARAM_OUT, index=False)
    print(f"参数已保存: {PARAM_OUT}")

    # --- [创新点绘图] 物理规律平稳性验证 (Train vs Val vs Test) ---
    plt.figure(figsize=(10, 7))
    
    # 1. 绘制训练集 (灰色背景)
    plt.scatter(x_train, y_train, alpha=0.05, s=2, color='gray', label='Train (Seasonally Mixed)')
    
    # 2. 绘制验证集 (绿色) - 检查是否覆盖全域
    plt.scatter(x_val, y_val, alpha=0.1, s=2, color='limegreen', label='Val (Held-out Blocks)')

    # 3. 绘制测试集 (蓝色) - 检查是否覆盖全域
    plt.scatter(x_test, y_test, alpha=0.1, s=2, color='dodgerblue', label='Test (Held-out Blocks)')
    
    # 4. 绘制拟合曲线
    # x_line = np.linspace(0.001, 0.9, 100)
    # y_line = power_law_model(x_line, *popt)
    # plt.plot(x_line, y_line, 'r-', linewidth=3, label=f'Fit: $I_F={a:.2f}P^{{{beta:.2f}}}+{c:.2f}$')
    
    plt.title('Stationarity Check: Interleaved Split', fontsize=14)
    plt.xlabel('Physical Baseline Power $P_{phys}$ (p.u.)', fontsize=12)
    plt.ylabel('Fluctuation Intensity $I_F$', fontsize=12)
    plt.legend(loc='upper right', markerscale=3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 2.5)
    
    out_png = OUT / "stationarity_check_interleaved.png"
    plt.savefig(out_png, dpi=150)
    print(f"验证图表已保存: {out_png}")
    
    # --- 保存 Origin 数据 ---
    pd.DataFrame({'P_train': x_train, 'I_F_train': y_train}).to_csv(OUT/"origin_train_mixed.csv", index=False)
    pd.DataFrame({'P_val': x_val,     'I_F_val': y_val}).to_csv(OUT/"origin_val_mixed.csv", index=False)
    pd.DataFrame({'P_test': x_test,   'I_F_test': y_test}).to_csv(OUT/"origin_test_mixed.csv", index=False)
    print("Origin 数据导出完成。")

except Exception as e:
    print(f"拟合失败: {e}")
    import traceback
    traceback.print_exc()

# --- 7. 保存分析数据 ---
out_csv = OUT / "phys_baseline_analysis_data.csv"
df_analysis.to_csv(out_csv)
print(f"分析数据已保存至: {out_csv}")