# -*- coding: utf-8 -*-
# 最终版本V5.1：已集成30分钟时间戳修正功能

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib
from pathlib import Path
from scipy.optimize import curve_fit

# ===================================================================
# 【重要】参数配置
# ===================================================================
# 1. 光伏电站与数据文件信息
LATITUDE = 34.05
LONGITUDE = -118.24
ALTITUDE = 71
TIMEZONE = "America/Los_Angeles"

# 2. 您的数据文件路径和额定功率
INP = Path("outputs/clean")
OUT = Path("outputs/pv_power_law_analysis"); OUT.mkdir(parents=True, exist_ok=True)
P_RATED_KW = 30.1

# 3. 分析参数
TIME_RESOLUTION = '15T' 

# 4. 【新增的时间修正参数】
# 根据之前的诊断，设定实际功率数据需要修正的分钟数
# 正数表示 P_actual 滞后 (峰值晚于晴空模型)，需要向前修正
LAG_TO_CORRECT_MINUTES = 0
# ===================================================================

# --- 1. 数据加载与预处理 ---
print("--- 步骤1：加载原始数据 ---")
def read_csv_tz(path_csv: Path, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv, index_col=0)
    idx = pd.to_datetime(df.index, utc=True).tz_convert(tz)
    df.index = idx
    return df

# 始终读取最原始、未经修正的文件
pv_1m = read_csv_tz(INP/"pv_1min_clean.csv", TIMEZONE)
pcol = next((c for c in pv_1m.columns if c.lower() in ["p_kw","power","power_kw"]), pv_1m.columns[0])
P_actual_kw = pd.to_numeric(pv_1m[pcol], errors="coerce").fillna(0.0).clip(lower=0.0)
print("步骤1：已加载实际光伏功率数据。")

# --- 1b. 【新增的时间戳修正步骤】 ---
if LAG_TO_CORRECT_MINUTES != 0:
    print(f"--- 步骤1b：对实际功率数据进行 {LAG_TO_CORRECT_MINUTES} 分钟的时间戳修正 ---")
    time_shift = pd.to_timedelta(LAG_TO_CORRECT_MINUTES, unit='m')
    P_actual_kw.index = P_actual_kw.index - time_shift
else:
    print("--- 步骤1b：无需进行时间戳修正 ---")

# --- 2. 【已修正】生成晴空功率 (采用更稳健的线性回归标定法) ---
print("--- 步骤2：正在生成晴空功率 (鲁棒标定法) ---")
location = pvlib.location.Location(latitude=LATITUDE, longitude=LONGITUDE, altitude=ALTITUDE, tz=TIMEZONE)
# 注意：现在P_actual_kw的时间戳已经是修正过的了
clearsky_ghi = location.get_clearsky(P_actual_kw.index)['ghi']
is_clear = pvlib.clearsky.detect_clearsky(P_actual_kw, clearsky_ghi, P_actual_kw.index)
print(f"在数据中检测到 {is_clear.sum()} 个晴空数据点。")
clearsky_points_actual_power = P_actual_kw[is_clear]
clearsky_points_ghi = clearsky_ghi[is_clear]
if not clearsky_points_actual_power.empty:
    x = clearsky_points_ghi.values.reshape(-1, 1)
    y = clearsky_points_actual_power.values
    xtx = np.dot(x.T, x)
    xty = np.dot(x.T, y)
    scaling_factor = (xty / xtx)[0, 0] if xtx[0, 0] > 0 else 0
else:
    print("警告：未检测到足够的晴空数据点，退回使用峰值标定法。")
    peak_actual_power = P_actual_kw.max()
    peak_ghi_clearsky = clearsky_ghi.max()
    scaling_factor = peak_actual_power / peak_ghi_clearsky if peak_ghi_clearsky > 0 else 0
P_clearsky_kw = (clearsky_ghi * scaling_factor).clip(lower=0)
print(f"通过线性回归确定的最佳缩放系数为: {scaling_factor:.4f}。")

# --- 3. 计算“云瞬变” (定义光伏的“波动分量”) ---
# 这里的 P_actual_kw 已经是修正过时间戳的
cloud_transient = (P_actual_kw - P_clearsky_kw)
cloud_transient[P_clearsky_kw < 0.001 * P_RATED_KW] = 0
print("步骤3：已计算“云瞬变”作为“波动分量”。")

# --- 4. 【已优化】计算光伏的“波动强度 I_F_PV” ---
# print(f"--- 步骤4：以 {TIME_RESOLUTION} 窗口计算光伏波动强度 ---")
# sigma_cloud = cloud_transient.resample(TIME_RESOLUTION).std().dropna()
# px_clearsky = P_clearsky_kw.resample(TIME_RESOLUTION).mean().dropna()
# df_analysis = pd.DataFrame({
#     'Px_clearsky': px_clearsky,
#     'sigma_cloud': sigma_cloud
# }).dropna()
# df_analysis = df_analysis[df_analysis['Px_clearsky'] > 0.01 * P_RATED_KW]
# df_analysis['I_F_PV'] = df_analysis['sigma_cloud'] / df_analysis['Px_clearsky']
# print(f"步骤4：已计算出每个 {TIME_RESOLUTION} 窗口的波动强度 I_F_PV。")
# --- 4. 以窗口计算“总波动强度”（严格按公式） ---
print(f"--- 步骤4：以 {TIME_RESOLUTION} 窗口计算光伏总波动强度 ---")

df_1m = pd.DataFrame({
    'P': P_actual_kw,
    'P_clear': P_clearsky_kw
})

# def sigma_total(group):
#     """
#     计算实际功率 P 相对于窗口内晴空功率均值 Pst 的均方根偏差，
#     采用 ddof=1 (样本标准差) 的无偏估计。
#     """
#     # P_st(j): 窗口内晴空功率的平均值，这是一个固定的参考点
#     Pst = group['P_clear'].mean()
    
#     # n: 窗口内的数据点数量
#     n = len(group)
    
#     # 对于样本估计，至少需要2个数据点才能计算
#     if n < 2:
#         return np.nan
        
#     # 计算 (P(i) - Pst)^2 的总和
#     sum_of_squares = ((group['P'] - Pst)**2).sum()
    
#     # 【核心修改】
#     # 使用 n-1 作为分母，这对应于 ddof=1
#     # 这被称为“贝塞尔校正” (Bessel's correction)
#     variance_unbiased = sum_of_squares / (n - 1)
    
#     # 返回其平方根
#     return np.sqrt(variance_unbiased)
def sigma_total(group):
    """
    计算每个窗口内“云瞬变”分量的标准差。
    这对应于：std(P_actual - P_clearsky)
    """
    # 1. 首先，在窗口内的每个时刻，计算实际功率与实时晴空功率的差值
    fluctuation = group['P'] - group['P_clear']
    
    # 2. 然后，对这个差值序列（即波动分量）求标准差
    # pandas 的 .std() 默认使用 ddof=1，这是样本标准差的无偏估计，是标准做法。
    return fluctuation.std()

# 这里得到的是 Series，所以用 rename('新名称')
sigma_pv  = df_1m.resample(TIME_RESOLUTION).apply(sigma_total).rename('sigma_PV')
Pst_mean = df_1m['P_clear'].resample(TIME_RESOLUTION).mean().rename('Pst')

df_analysis = pd.concat([Pst_mean, sigma_pv], axis=1).dropna()
df_analysis = df_analysis[df_analysis['Pst'] > 0.01 * P_RATED_KW]
df_analysis['I_F_PV'] = df_analysis['sigma_PV'] / df_analysis['Pst']
df_analysis = df_analysis.rename(columns={'Pst': 'Px_clearsky', 'sigma_PV': 'sigma_cloud'})
# --- 5. 拟合解析表达式 (三参数幂律模型) ---
def power_law_model(p, a, beta, c):
    return a * np.power(p, beta) + c
x_data = df_analysis['Px_clearsky'] / P_RATED_KW
y_data = df_analysis['I_F_PV']
valid_indices = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0)
x_fit, y_fit = x_data[valid_indices], y_data[valid_indices]

try:
    params, covariance = curve_fit(power_law_model, x_fit, y_fit, p0=[0.1, -0.5, 0.1], maxfev=5000)
    a, beta, c = params
    print("\n步骤5：成功拟合三参数幂律模型！")
    print("="*40)
    print(f"         {TIME_RESOLUTION} 窗口下的光伏波动强度解析表达式:")
    print(f"         I_F = {a:.4f} * P_st^{beta:.4f} + {c:.4f}")
    print("="*40)

    # --- 6. 可视化拟合结果 ---
    plt.figure(figsize=(10, 7))
    plt.scatter(x_fit, y_fit, alpha=0.3, label=f'实际数据点 ({TIME_RESOLUTION})')
    x_curve = np.linspace(x_fit.min(), x_fit.max(), 200)
    y_curve = power_law_model(x_curve, a, beta, c)
    plt.plot(x_curve, y_curve, color='red', linewidth=2.5, label='幂律模型拟合曲线')
    plt.title(f'光伏波动强度 vs. 晴空平稳分量 ({TIME_RESOLUTION} 窗口)', fontsize=16)
    plt.xlabel('晴空平稳分量 P_st (p.u.)', fontsize=12)
    plt.ylabel('光伏波动强度 I_F', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=0)
    plt.ylim(bottom=-0.1)
    
    output_filename = f"pv_fluctuation_intensity_fit_{TIME_RESOLUTION}.png"
    plt.savefig(OUT/output_filename, dpi=150)
    plt.show()

except RuntimeError:
    print("\n[ERROR] 曲线拟合失败。可能是数据量太少或数据分布不适合幂律模型。")

# --- 7. 【已修正】保存关键数据 ---
df_to_save_1min = pd.DataFrame({
    'cloud_transient': cloud_transient,
    'P_clearsky_kw': P_clearsky_kw
})
output_path_1min = OUT / "components_for_wavelet_analysis.csv"
df_to_save_1min.to_csv(output_path_1min)
print(f"\n步骤7：已保存1分钟波动与平稳分量至: {output_path_1min}")

df_resampled_stats = df_analysis.copy()
output_path_resampled = OUT / f"statistical_analysis_data_{TIME_RESOLUTION}.csv"
df_resampled_stats.to_csv(output_path_resampled)
print(f"步骤7：已保存 {TIME_RESOLUTION} 分辨率的统计分析数据至: {output_path_resampled}")