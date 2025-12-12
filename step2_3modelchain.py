# -*- coding: utf-8 -*-
# 版本V6.2：固定物理参数 (无标定) + 物理清洗 + Interleaved Split + 双对数图
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
LATITUDE = 37.427963
LONGITUDE = -122.154785
ALTITUDE = 71
TIMEZONE = "Etc/GMT+8" 

# 路径配置
CURRENT_DIR = Path(__file__).parent.absolute()
INP = CURRENT_DIR / "rawdata"
FILE_NAME = INP / "merged_2018_2019_5min_UTC_cleaned.csv" # 请确认文件名
OUT = CURRENT_DIR / "outputs" / "Scatter_FixedParams"
OUT.mkdir(parents=True, exist_ok=True)

# 分析参数
P_RATED_KW = 30.1
TIME_RESOLUTION = '15min'

# 【关键修改】使用固定参数，不再进行自动标定
FIXED_TILT = 22.5      
FIXED_AZIMUTH = 195    

print(f"\n[调试信息] 开始运行 V6.2 (固定参数版)")
print(f"设定参数: Tilt={FIXED_TILT}, Azimuth={FIXED_AZIMUTH}")
print(f"读取文件: {FILE_NAME}")
print(f"="*30 + "\n")

# ===================================================================
# --- 模块1：数据加载 ---
# ===================================================================
def read_csv_tz(path_csv: Path, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    pcol = next((c for c in df.columns if c in ['Target_Power', 'power_kw', 'p_kw', 'Power(kW)']), None)
    time_col = next((c for c in df.columns if 'time' in c.lower() or 'ts' in c.lower() or 'Date' in c), df.columns[0])
    
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(tz)
    df.set_index(time_col, inplace=True)
    return df, pcol

full_path = INP / FILE_NAME
if not full_path.exists():
    print(f"❌ 错误: 文件 {FILE_NAME} 不存在")
    exit()

df_raw, p_col_name = read_csv_tz(full_path, TIMEZONE)
P_actual_kw = pd.to_numeric(df_raw[p_col_name], errors="coerce").fillna(0.0).clip(lower=0.0)

# 气象数据兼容处理
if 'Obs_Temp' in df_raw.columns: temp_air = df_raw['Obs_Temp']
elif 'temp_air' in df_raw.columns: temp_air = df_raw['temp_air']
else: temp_air = pd.Series(20, index=df_raw.index)

if 'NWP_Wind' in df_raw.columns: wind_speed = df_raw['NWP_Wind']
elif 'wind_speed' in df_raw.columns: wind_speed = df_raw['wind_speed']
else: wind_speed = pd.Series(1, index=df_raw.index)

# ===================================================================
# --- 模块2：(已跳过) 自动光学标定 ---
# ===================================================================
# 此处逻辑已移除，直接使用配置区的固定参数
BEST_TILT = FIXED_TILT
BEST_AZIMUTH = FIXED_AZIMUTH

# 初始化位置对象
location_obj = pvlib.location.Location(latitude=LATITUDE, longitude=LONGITUDE, tz=TIMEZONE, altitude=ALTITUDE)

# ===================================================================
# --- 模块3：使用固定参数运行物理模型 ---
# ===================================================================
print("--- 步骤3：基于固定参数计算物理基准 ---")

system = PVSystem(
    surface_tilt=BEST_TILT,
    surface_azimuth=BEST_AZIMUTH,
    module_parameters={'pdc0': P_RATED_KW * 1000, 'gamma_pdc': -0.003},
    inverter_parameters={'pdc0': P_RATED_KW * 1000, 'eta_inv_nom': 0.95},
    temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
)

mc = ModelChain(system, location_obj, aoi_model='physical', spectral_model='no_loss')

# 全量计算
cs = location_obj.get_clearsky(df_raw.index)
weather_clearsky = pd.DataFrame({
    'ghi': cs['ghi'], 'dni': cs['dni'], 'dhi': cs['dhi'],
    'temp_air': temp_air, 'wind_speed': wind_speed
})

mc.run_model(weather_clearsky)
P_phys_clearsky = mc.results.ac.fillna(0).clip(lower=0) / 1000.0

# ===================================================================
# --- 模块4：高级物理清洗 (含 Rolling Fix) ---
# ===================================================================
print("--- 步骤4：计算指标与高级清洗 ---")

# 4.1 计算统计量
df_calc = pd.DataFrame({'P_actual': P_actual_kw, 'P_clear': P_phys_clearsky})

# [重要修复] 使用 rolling 计算波动，防止 resample 导致 NaN
# sigma_pv = df_calc['P_actual'].rolling(window=3, center=True, min_periods=2).std().rename('sigma_PV')
sigma_pv = df_calc['P_actual'].resample(TIME_RESOLUTION).std().rename('sigma_PV')
Pst_mean = df_calc['P_clear'].resample(TIME_RESOLUTION).mean().rename('Px_clearsky')
P_act_mean = df_calc['P_actual'].resample(TIME_RESOLUTION).mean().rename('P_actual_mean') 

# 获取太阳高度角
# 确保索引对齐
common_idx = Pst_mean.index.intersection(location_obj.get_solarposition(Pst_mean.index).index)
solpos = location_obj.get_solarposition(common_idx)
zenith = solpos['zenith']

df_analysis = pd.concat([Pst_mean, sigma_pv, P_act_mean, zenith], axis=1).dropna()

# 4.2 定义清洗过滤器
print(f"   原始数据点数: {len(df_analysis)}")

if len(df_analysis) == 0:
    print("❌ 错误: 数据交集为空，请检查时间索引。")
    exit()

# Filter A: 夜间过滤
mask_day = df_analysis['Px_clearsky'] > 0.02 * P_RATED_KW

# Filter B: 太阳高度角过滤
mask_zenith = df_analysis['zenith'] < 80 

# Filter C: 削峰(Clipping) 过滤
mask_no_clipping = df_analysis['P_actual_mean'] < (0.98 * P_RATED_KW)

# Filter D: 严重遮挡过滤
mask_no_shading = ~((df_analysis['Px_clearsky'] > 0.1 * P_RATED_KW) & (df_analysis['P_actual_mean'] < 0.01 * P_RATED_KW))

# 综合 Mask
final_mask = mask_day & mask_zenith & mask_no_clipping & mask_no_shading
df_clean = df_analysis[final_mask].copy()

print(f"   清洗后数据点数: {len(df_clean)} (移除率: {(1 - len(df_clean)/len(df_analysis))*100:.1f}%)")

# 计算波动强度 I_F (加微小量防止除零)
df_clean['I_F_PV'] = df_clean['sigma_PV'] / (df_clean['Px_clearsky'] + 1e-6)

# ==============================================================================
# --- 步骤 5：交互式手动调参 (Interactive Manual Tuning) ---
# ==============================================================================
print("--- 步骤5：启动交互式调参窗口 ---")
print("   提示：弹出的窗口中，请用鼠标拖动滑块来调整 a, beta, gamma")

from matplotlib.widgets import Slider, Button

# 1. 准备数据 (沿用之前的清洗逻辑)
start_date = df_clean.index[0]
days_diff = (df_clean.index - start_date).days
biweek_idx = days_diff // 14
cycle_idx = biweek_idx % 6
train_mask = np.isin(cycle_idx, [0, 1, 2, 3])
df_train = df_clean[train_mask]

def get_xy(df):
    x = df['Px_clearsky'] / P_RATED_KW
    y = df['I_F_PV']
    # 严格过滤，防止 log(负数)
    mask = (x > 0.02) & (x < 0.999) & (y > 0.001) & np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]

x_train, y_train = get_xy(df_train)

# 2. 定义模型
def cutoff_power_law(p, a, beta, gamma):
    return a * np.power(p, beta) * np.power(1 - p, gamma)

# 3. 初始化绘图
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.25) # 底部留出空间给滑块

# 绘制背景数据 (灰色)
ax.scatter(x_train, y_train, alpha=0.1, s=5, c='gray', label='Train Data', edgecolors='none')

# 绘制初始曲线
p_space = np.geomspace(x_train.min(), x_train.max(), 500)
# 初始参数猜测
init_a = 0.1
init_beta = -0.6
init_gamma = 1.0

l, = ax.plot(p_space, cutoff_power_law(p_space, init_a, init_beta, init_gamma), 
             lw=3, color='red', label='Manual Fit')

# 设置对数坐标
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Physical Baseline Power $P_{phys}$ (p.u.) [Log Scale]', fontsize=12)
ax.set_ylabel(r'Fluctuation Intensity $I_F$ [Log Scale]', fontsize=12)
ax.set_title('Interactive Tuning: Cut-off Power Law\n$I_F = a \cdot P^{\\beta} \cdot (1-P)^{\\gamma}$', fontsize=14)
ax.set_xlim(0.02, 1.05)
ax.set_ylim(0.005, 5.0)
ax.grid(True, which="both", ls="-", alpha=0.3)
ax.legend(loc='upper right')

# 4. 添加滑块 (Sliders)
# 位置: [left, bottom, width, height]
axcolor = 'lightgoldenrodyellow'
ax_a     = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_beta  = plt.axes([0.2, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_gamma = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)

# 定义滑块范围
s_a = Slider(ax_a, 'a (Scale)', 0.01, 0.5, valinit=init_a)
s_beta = Slider(ax_beta, 'Beta (Turbulence)', -2.0, 0.0, valinit=init_beta)
s_gamma = Slider(ax_gamma, 'Gamma (Saturation)', 0.0, 10.0, valinit=init_gamma)

# 5. 更新函数
def update(val):
    a = s_a.val
    beta = s_beta.val
    gamma = s_gamma.val
    
    # 重新计算曲线
    y_new = cutoff_power_law(p_space, a, beta, gamma)
    l.set_ydata(y_new)
    
    # 更新标题显示数值
    ax.set_title(f'Interactive Tuning: Cut-off Power Law\n$a={a:.3f}, \\beta={beta:.3f}, \\gamma={gamma:.3f}$', fontsize=14)
    fig.canvas.draw_idle()

# 绑定事件
s_a.on_changed(update)
s_beta.on_changed(update)
s_gamma.on_changed(update)

# 添加重置按钮
resetax = plt.axes([0.8, 0.01, 0.1, 0.03])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    s_a.reset()
    s_beta.reset()
    s_gamma.reset()
button.on_clicked(reset)

print("   >>> 请在弹出的窗口中进行操作...")
plt.show()

# --- 提示：调完之后，手动记下你满意的参数，填入下方变量用于保存 ---
print("\n" + "="*50)
print("【调参结束后操作】")
print("请将你满意的参数手动填入代码中，以生成最终保存的图片：")
print("FINAL_A = ...")
print("FINAL_BETA = ...")
print("FINAL_GAMMA = ...")
print("="*50 + "\n")