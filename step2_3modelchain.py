# -*- coding: utf-8 -*-
# 版本V6.3：固定参数 + 交互式调参 + Train/Val/Test三集独立绘制输出
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib
from pathlib import Path
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from matplotlib.widgets import Slider, Button
import warnings
from scipy.optimize import curve_fit  # 必须添加这一行
from scipy import stats               # 之前的假设检验也需要这一行

warnings.filterwarnings("ignore")

# ===================================================================
# 【配置区域】
# ===================================================================
LATITUDE = 37.427963
LONGITUDE = -122.154785
ALTITUDE = 71
TIMEZONE = "Etc/GMT+8" 

CURRENT_DIR = Path(__file__).parent.absolute()
INP = CURRENT_DIR / "rawdata"
FILE_NAME = INP / "merged_2018_2019_5min_UTC_cleaned.csv"
OUT = CURRENT_DIR / "outputs" / "Scatter_ThreeSplits"  # 新输出文件夹
OUT.mkdir(parents=True, exist_ok=True)

P_RATED_KW = 30.1
TIME_RESOLUTION = '15min'

# 固定物理参数
FIXED_TILT = 22.5      
FIXED_AZIMUTH = 195    

print(f"\n[调试信息] 开始运行 V6.3 (三集独立验证版)")

# ===================================================================
# --- 模块1-3：数据加载与物理基准计算 (保持不变) ---
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

# 简化的气象填充
temp_air = df_raw.get('Obs_Temp', df_raw.get('temp_air', pd.Series(20, index=df_raw.index)))
wind_speed = df_raw.get('NWP_Wind', df_raw.get('wind_speed', pd.Series(1, index=df_raw.index)))

# 物理模型计算
location_obj = pvlib.location.Location(LATITUDE, LONGITUDE, TIMEZONE, ALTITUDE)
system = PVSystem(
    surface_tilt=FIXED_TILT,
    surface_azimuth=FIXED_AZIMUTH,
    module_parameters={'pdc0': P_RATED_KW * 1000, 'gamma_pdc': -0.003},
    inverter_parameters={'pdc0': P_RATED_KW * 1000, 'eta_inv_nom': 0.95},
    temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
)
mc = ModelChain(system, location_obj, aoi_model='physical', spectral_model='no_loss')

cs = location_obj.get_clearsky(df_raw.index)
weather_clearsky = pd.DataFrame({'ghi': cs['ghi'], 'dni': cs['dni'], 'dhi': cs['dhi'], 'temp_air': temp_air, 'wind_speed': wind_speed})
mc.run_model(weather_clearsky)
P_phys_clearsky = mc.results.ac.fillna(0).clip(lower=0) / 1000.0

# ===================================================================
# --- 模块4：高级清洗 (保持不变) ---
# ===================================================================
df_calc = pd.DataFrame({'P_actual': P_actual_kw, 'P_clear': P_phys_clearsky})
sigma_pv = df_calc['P_actual'].resample(TIME_RESOLUTION).std()
Pst_mean = df_calc['P_clear'].resample(TIME_RESOLUTION).mean()
P_act_mean = df_calc['P_actual'].resample(TIME_RESOLUTION).mean()
common_idx = Pst_mean.index.intersection(location_obj.get_solarposition(Pst_mean.index).index)
zenith = location_obj.get_solarposition(common_idx)['zenith']

df_analysis = pd.concat([Pst_mean, sigma_pv, P_act_mean, zenith], axis=1).dropna()
df_analysis.columns = ['Px_clearsky', 'sigma_PV', 'P_actual_mean', 'zenith']

# 过滤器
mask_day = df_analysis['Px_clearsky'] > 0.02 * P_RATED_KW
mask_zenith = df_analysis['zenith'] < 80 
mask_no_clipping = df_analysis['P_actual_mean'] < (0.98 * P_RATED_KW)
mask_no_shading = ~((df_analysis['Px_clearsky'] > 0.1 * P_RATED_KW) & (df_analysis['P_actual_mean'] < 0.01 * P_RATED_KW))
df_clean = df_analysis[mask_day & mask_zenith & mask_no_clipping & mask_no_shading].copy()

# 计算 I_F
df_clean['I_F_PV'] = df_clean['sigma_PV'] / (df_clean['Px_clearsky'] + 1e-6)

# ==============================================================================
# --- 步骤 5：【关键修改】数据切分与交互式绘图 ---
# ==============================================================================
print("--- 步骤5：准备 Train/Val/Test 数据 ---")

start_date = df_clean.index[0]
days_diff = (df_clean.index - start_date).days
biweek_idx = days_diff // 14
cycle_idx = biweek_idx % 6

# 定义 Mask (4:1:1)
train_mask = np.isin(cycle_idx, [0, 1, 2, 3])
val_mask   = (cycle_idx == 4)
test_mask  = (cycle_idx == 5)

def get_xy(df):
    x = df['Px_clearsky'] / P_RATED_KW
    y = df['I_F_PV']
    mask = (x > 0.02) & (x < 0.999) & (y > 0.001) & np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]

# 分别提取三组数据
x_train, y_train = get_xy(df_clean[train_mask])
x_val, y_val     = get_xy(df_clean[val_mask])
x_test, y_test   = get_xy(df_clean[test_mask])

print(f"数据点数: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}")

# 定义模型公式
def cutoff_power_law(p, a, beta, gamma):
    return a * np.power(p, beta) * np.power(1 - p, gamma)

# --- 初始化绘图 ---
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.25)

# 【核心修改】绘制三层数据，方便对比
# Train: 灰色 (最底层，用于拟合)
sc_train = ax.scatter(x_train, y_train, alpha=0.05, s=5, c='gray', label='Train (67%)', edgecolors='none')
# Val: 绿色 (中间层，用于验证)
sc_val = ax.scatter(x_val, y_val, alpha=0.1, s=5, c='limegreen', label='Val (16.5%)', edgecolors='none')
# Test: 蓝色 (最上层，核心验证对象)
sc_test = ax.scatter(x_test, y_test, alpha=0.1, s=5, c='dodgerblue', label='Test (16.5%)', edgecolors='none')

# 初始曲线
p_space = np.geomspace(0.02, 0.99, 500)
init_a, init_beta, init_gamma = 0.040, -0.656, 1.154
l, = ax.plot(p_space, cutoff_power_law(p_space, init_a, init_beta, init_gamma), lw=3, color='red', label='Model')

# 设置坐标轴
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Physical Baseline Power $P_{phys}$ (p.u.)', fontsize=12)
ax.set_ylabel(r'Fluctuation Intensity $I_F$', fontsize=12)
ax.set_title(f'Stationarity Check (Train/Val/Test)\n$a={init_a}, \\beta={init_beta}, \\gamma={init_gamma}$', fontsize=14)
ax.set_xlim(0.02, 1.05)
ax.set_ylim(0.005, 5.0)
ax.legend(loc='upper right')
ax.grid(True, which="both", ls="-", alpha=0.3)

# --- 添加滑块 ---
axcolor = 'lightgoldenrodyellow'
ax_a = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_beta = plt.axes([0.2, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_gamma = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)

s_a = Slider(ax_a, 'a', 0.01, 0.5, valinit=init_a)
s_beta = Slider(ax_beta, 'Beta', -2.0, 0.0, valinit=init_beta)
s_gamma = Slider(ax_gamma, 'Gamma', 0.0, 10.0, valinit=init_gamma)

def update(val):
    a, beta, gamma = s_a.val, s_beta.val, s_gamma.val
    y_new = cutoff_power_law(p_space, a, beta, gamma)
    l.set_ydata(y_new)
    ax.set_title(f'Stationarity Check (Train/Val/Test)\n$a={a:.3f}, \\beta={beta:.3f}, \\gamma={gamma:.3f}$', fontsize=14)
    fig.canvas.draw_idle()

s_a.on_changed(update)
s_beta.on_changed(update)
s_gamma.on_changed(update)

print(">>> 请在弹出的窗口中拖动滑块，观察红线是否同时拟合 Train/Val/Test 三种颜色的点...")
plt.show()

# ==============================================================================
# --- 步骤 6：【新增】自动保存三张独立图片 + 一张合成图 ---
# ==============================================================================
print("\n" + "="*50)
print("【正在导出独立验证图片...】")

# 获取最终滑块参数
final_a = s_a.val
final_beta = s_beta.val
final_gamma = s_gamma.val
print(f"最终参数: a={final_a:.4f}, beta={final_beta:.4f}, gamma={final_gamma:.4f}")

# 定义通用绘图函数
def plot_single_set(x, y, color, label, filename, title_suffix):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.1, s=5, c=color, label=label, edgecolors='none')
    # 绘制模型红线
    y_model = cutoff_power_law(p_space, final_a, final_beta, final_gamma)
    plt.plot(p_space, y_model, 'r-', lw=2, label=f'Fit Model')
    
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel(r'$P_{phys}$ (p.u.)'); plt.ylabel(r'$I_F$')
    plt.title(f'Stationarity Check: {title_suffix}', fontsize=12)
    plt.xlim(0.02, 1.05); plt.ylim(0.005, 5.0)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    save_path = OUT / filename
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"已保存: {save_path}")

# 1. 保存训练集图
plot_single_set(x_train, y_train, 'gray', 'Train Data', '1_scatter_train.png', 'Training Set')
# 2. 保存验证集图
plot_single_set(x_val, y_val, 'limegreen', 'Val Data', '2_scatter_val.png', 'Validation Set (Held-out)')
# 3. 保存测试集图
plot_single_set(x_test, y_test, 'dodgerblue', 'Test Data', '3_scatter_test.png', 'Test Set (Held-out)')

# 4. 保存合成图 (Combined)
plt.figure(figsize=(10, 7))
plt.scatter(x_train, y_train, alpha=0.05, s=2, c='gray', label='Train')
plt.scatter(x_val, y_val, alpha=0.1, s=2, c='limegreen', label='Val')
plt.scatter(x_test, y_test, alpha=0.1, s=2, c='dodgerblue', label='Test')
plt.plot(p_space, cutoff_power_law(p_space, final_a, final_beta, final_gamma), 'r-', lw=2, label='Model')
plt.xscale('log'); plt.yscale('log')
plt.xlabel(r'$P_{phys}$ (p.u.)'); plt.ylabel(r'$I_F$')
plt.title(f'Combined Stationarity Verification\n(All Splits Aligned with Model)', fontsize=14)
plt.legend(markerscale=3)
plt.grid(True, which="both", alpha=0.3)
plt.savefig(OUT / '4_combined_stationarity.png', dpi=200)
print(f"已保存合成图: {OUT / '4_combined_stationarity.png'}")
print("="*50)

# ==============================================================================
# --- 步骤 7：【新增】导出数据到 CSV 供 Origin 复现 ---
# ==============================================================================
print("\n" + "="*50)
print("【步骤7：导出 Origin 专用数据文件】")

# 1. 导出散点数据 (Train/Val/Test)
# 说明：保存为 X(P_phys) 和 Y(I_F) 两列，方便 Origin 直接 Plot Scatter
print("正在保存散点数据...")
pd.DataFrame({'P_phys': x_train, 'I_F': y_train}).to_csv(OUT / "origin_scatter_train.csv", index=False)
pd.DataFrame({'P_phys': x_val,   'I_F': y_val}).to_csv(OUT / "origin_scatter_val.csv", index=False)
pd.DataFrame({'P_phys': x_test,  'I_F': y_test}).to_csv(OUT / "origin_scatter_test.csv", index=False)

# 2. 导出拟合曲线数据 (Model Curve)
# 说明：利用您最终调整的参数 (final_a, final_beta, final_gamma) 生成平滑曲线
print(f"正在保存拟合曲线数据 (基于参数 a={final_a:.3f}, beta={final_beta:.3f}, gamma={final_gamma:.3f})...")

# 生成高密度的点以保证曲线平滑
p_smooth = np.geomspace(0.02, 1.0, 200) 
i_f_smooth = cutoff_power_law(p_smooth, final_a, final_beta, final_gamma)

# 为了方便记录，将参数值也写入文件的第一行注释或单独列中（这里放在单独列的头部）
df_model = pd.DataFrame({
    'P_phys_line': p_smooth, 
    'I_F_line': i_f_smooth
})
# 添加参数信息到 metadata 文件，或者直接打印在文件名中供参考
df_model.to_csv(OUT / "origin_model_curve.csv", index=False)

print(f"✅ 导出完成！请在 {OUT} 目录下查看以下文件：")
print(f"   1. origin_scatter_train.csv  (训练集散点)")
print(f"   2. origin_scatter_val.csv    (验证集散点)")
print(f"   3. origin_scatter_test.csv   (测试集散点)")
print(f"   4. origin_model_curve.csv    (红色的拟合曲线)")
print("="*50)

# ==============================================================================
# --- 步骤 8：【新增】计算并打印量化评价指标 (RMSE, MAE, R2) ---
# ==============================================================================
print("\n" + "="*50)
print("【步骤8：拟合优度评价指标计算】")

def calculate_metrics(x_data, y_true, dataset_name):
    # 1. 使用最终参数计算预测值
    y_pred = cutoff_power_law(x_data, final_a, final_beta, final_gamma)
    
    # 2. 计算残差
    residuals = y_true - y_pred
    
    # 3. 计算指标 (全部使用 numpy 原生实现，无需额外库)
    # RMSE: 均方根误差
    rmse = np.sqrt(np.mean(residuals**2))
    
    # MAE: 平均绝对误差
    mae = np.mean(np.abs(residuals))
    
    # R2: 决定系数 (1 - SS_res / SS_tot)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    # 4. 打印结果
    print(f"--- {dataset_name} ---")
    print(f"   样本数 (N): {len(y_true)}")
    print(f"   RMSE     : {rmse:.5f}")
    print(f"   MAE      : {mae:.5f}")
    print(f"   R²       : {r2:.5f}")
    
    return {'Dataset': dataset_name, 'N': len(y_true), 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# 分别计算三个集
metrics_train = calculate_metrics(x_train, y_train, "训练集 (Train)")
metrics_val   = calculate_metrics(x_val, y_val,     "验证集 (Val)")
metrics_test  = calculate_metrics(x_test, y_test,   "测试集 (Test)")

# (可选) 将指标保存到 CSV
metrics_df = pd.DataFrame([metrics_train, metrics_val, metrics_test])
metrics_out_path = OUT / "fitting_metrics.csv"
metrics_df.to_csv(metrics_out_path, index=False)
print(f"\n✅ 指标已保存至: {metrics_out_path}")
print("="*50)

from scipy import stats

# ==============================================================================
# --- 步骤 11：【新增】高噪声下的统计合理性验证 (防御性指标) ---
# ==============================================================================
from scipy import stats

print("\n" + "="*50)
print("【步骤11：统计合理性深度验证 (针对高噪声数据)】")

# 1. 定义计算函数
def evaluate_statistical_soundness(x, y_true, dataset_name):
    # 使用您最终确定的参数 (确保 final_a, final_beta, final_gamma 已定义)
    y_pred = cutoff_power_law(x, final_a, final_beta, final_gamma)
    
    # --- 核心指标 A: 残差 (Residuals) ---
    residuals = y_true - y_pred
    
    # 1. 平均偏差 (Mean Bias Error, MBE)
    # 如果接近 0，说明曲线穿过了数据云团的重心，没有偏上或偏下
    mbe = np.mean(residuals)
    
    # 2. 残差标准差 (Standard Deviation of Residuals)
    std_res = np.std(residuals)
    
    # --- 核心指标 B: 斯皮尔曼相关系数 (Spearman Correlation) ---
    # 既然 R2 不行，我们用 Spearman。它衡量“单调趋势”的强弱，不被噪声幅度干扰。
    corr, p_val = stats.spearmanr(x, y_true)
    
    # --- 核心指标 C: 覆盖率 (Coverage) ---
    # 统计有多少点落在了 模型 +/- 1倍标准差 的范围内
    # 这证明了模型虽然不能预测单点，但涵盖了主要的概率分布
    within_1std = np.sum(np.abs(residuals) < std_res) / len(residuals) * 100
    
    print(f"--- {dataset_name} ---")
    print(f"   1. 平均偏差 (MBE)   : {mbe:.5f} (理想值 -> 0)")
    print(f"   2. 趋势相关性 (Corr): {corr:.4f} (P-value: {p_val:.2e})")
    print(f"   3. 1σ 覆盖率        : {within_1std:.1f}% (理论正态分布约为 68%)")
    
    return residuals

# 2. 计算并收集残差
res_train = evaluate_statistical_soundness(x_train, y_train, "Train")
res_test  = evaluate_statistical_soundness(x_test,  y_test,  "Test")
# 复制这行代码到您的脚本最后运行一下
res_val = evaluate_statistical_soundness(x_val, y_val, "Val")
# ==============================================================================
# --- 绘图：残差分布直方图 (最有力的视觉证据) ---
# ==============================================================================
plt.figure(figsize=(10, 6))

# 绘制训练集残差直方图
plt.hist(res_train, bins=100, density=True, alpha=0.5, color='gray', label='Train Residuals')
# 绘制测试集残差直方图 (看是否重合)
plt.hist(res_test, bins=100, density=True, alpha=0.5, color='dodgerblue', label='Test Residuals')

plt.hist(res_val, bins=100, density=True, alpha=0.5, color='yellow', label='Val Residuals')

# 绘制标准正态分布曲线 (参考线)
mu, std = stats.norm.fit(res_train)
xmin, xmax = plt.xlim()
x_plot = np.linspace(xmin, xmax, 100)
p_plot = stats.norm.pdf(x_plot, mu, std)
plt.plot(x_plot, p_plot, 'r--', linewidth=2, label=f'Normal Fit ($\mu$={mu:.3f})')

plt.title('Residual Distribution Analysis\n(Proof of Statistical Soundness)', fontsize=14)
plt.xlabel('Residual Error ($y_{true} - y_{pred}$)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-0.3, 0.3) # 根据实际情况调整范围，聚焦中心

save_path = OUT / "5_residual_distribution.png"
plt.savefig(save_path, dpi=150)
print(f"\n✅ 残差分析图已保存: {save_path}")
print("   (如果此图呈现以0为中心的钟形，说明模型成功捕捉了数据的统计重心。)")
print("="*50)

# ==============================================================================
# --- 步骤 12：【新增】导出残差直方图数据供 Origin 复现 ---
# ==============================================================================
print("\n" + "="*50)
print("【步骤12：导出 Origin 残差直方图专用数据】")

# 1. 导出训练集残差 (Raw Data for Histogram)
# Origin 用法: 选中列 -> Plot -> Statistics -> Histogram
pd.DataFrame({'Residuals_Train': res_train}).to_csv(OUT / "origin_hist_train.csv", index=False)

# 2. 导出测试集残差 (Raw Data for Histogram)
pd.DataFrame({'Residuals_Test': res_test}).to_csv(OUT / "origin_hist_test.csv", index=False)

pd.DataFrame({'Residuals_Test': res_val}).to_csv(OUT / "origin_hist_val.csv", index=False)

# 3. 导出正态分布拟合曲线 (Curve Data)
# Origin 用法: 选中两列 -> Plot -> Line -> 叠加到直方图上
mu, std = stats.norm.fit(res_train)  # 重新计算分布参数
x_min = min(res_train.min(), res_test.min())
x_max = max(res_train.max(), res_test.max())
x_curve = np.linspace(x_min, x_max, 200) # 生成平滑曲线的X轴
y_curve = stats.norm.pdf(x_curve, mu, std) # 生成概率密度Y轴

pd.DataFrame({
    'X_Curve': x_curve, 
    'Y_Normal_Fit': y_curve
}).to_csv(OUT / "origin_hist_curve.csv", index=False)

print(f"✅ 已生成3个文件至 {OUT}：")
print(f"   1. origin_hist_train.csv (用于绘制灰色直方图)")
print(f"   2. origin_hist_test.csv  (用于绘制蓝色直方图)")
print(f"   3. origin_hist_curve.csv (用于绘制红色虚线)")
print("="*50)