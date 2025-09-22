# -*- coding: utf-8 -*-
# 最终版本V4.2：集成了时间戳修正、修复了bug并清理了代码

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib
from pathlib import Path
from scipy.optimize import curve_fit
from scipy import stats
import pywt
import seaborn as sns

# ===================================================================
# 【重要】参数配置
# ===================================================================
# 1. 光伏电站与数据文件信息
LATITUDE = 34.05
LONGITUDE = -118.24
ALTITUDE = 71
TIMEZONE = "America/Los_Angeles"
INP = Path("outputs/clean")
OUT = Path("outputs/pv_power_law_analysis"); OUT.mkdir(parents=True, exist_ok=True)
P_RATED_KW = 30.1

# 2. 分析参数
TIME_RESOLUTION = '60T' # <<< 时间窗口

# 3. 【新增的时间修正参数】
LAG_TO_CORRECT_MINUTES = 0

# 4. 幂律模型函数 (仅用于低频分量)
def power_law_model(p, a, beta, c):
    return a * np.power(p, beta) + c
# ===================================================================

# --- 1. 数据加载与预处理 ---
print("--- 步骤1：加载原始数据 ---")
def read_csv_tz(path_csv: Path, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv, index_col=0)
    idx = pd.to_datetime(df.index, utc=True).tz_convert(tz)
    df.index = idx
    return df

pv_1m = read_csv_tz(INP/"pv_1min_clean.csv", TIMEZONE)
pcol = next((c for c in pv_1m.columns if c.lower() in ["p_kw","power","power_kw"]), pv_1m.columns[0])
P_actual_kw = pd.to_numeric(pv_1m[pcol], errors="coerce").fillna(0.0).clip(lower=0.0)

# --- 1b. 【新增】对实际功率数据进行时间戳修正 ---
if LAG_TO_CORRECT_MINUTES != 0:
    print(f"--- 步骤1b：对实际功率数据进行 {LAG_TO_CORRECT_MINUTES} 分钟的时间戳修正 ---")
    time_shift = pd.to_timedelta(LAG_TO_CORRECT_MINUTES, unit='m')
    P_actual_kw.index = P_actual_kw.index - time_shift
else:
    print("--- 步骤1b：无需进行时间戳修正 ---")

# --- 2. 计算平稳分量 (晴空功率) 与总波动分量 ---
# (这里替换回了更稳健的峰值标定法，因为线性回归法在修正时间戳后可能需要重新调试)
print("--- 步骤2：计算晴空功率与总波动 ---")
location = pvlib.location.Location(latitude=LATITUDE, longitude=LONGITUDE, altitude=ALTITUDE, tz=TIMEZONE)
clearsky = location.get_clearsky(P_actual_kw.index)
peak_actual_power = P_actual_kw.max()
peak_ghi_clearsky = clearsky['ghi'].max()
scaling_factor = peak_actual_power / peak_ghi_clearsky if peak_ghi_clearsky > 0 else 0
P_clearsky_kw = (clearsky['ghi'] * scaling_factor).clip(lower=0)
cloud_transient_total = (P_actual_kw - P_clearsky_kw)
cloud_transient_total[P_clearsky_kw < 0.001 * P_RATED_KW] = 0

# --- 3. 小波分解与选择性重构 ---
print("--- 步骤3：使用小波变换重构不同时间尺度的波动分量 ---")
def reconstruct_signal_at_scales(coeffs, levels_to_keep: list):
    recon_coeffs = [np.zeros_like(c) for c in coeffs]
    for level in levels_to_keep:
        if 0 < level <= len(coeffs) - 1:
            recon_coeffs[-level] = coeffs[-level]
    return pywt.waverec(recon_coeffs, 'db4')

wavelet = 'db4'
levels = 8
coeffs = pywt.wavedec(cloud_transient_total, wavelet, level=levels)
high_freq_levels, medium_freq_levels, low_freq_levels = [1, 2, 3], [4, 5], [6, 7, 8]
hf_volatility = reconstruct_signal_at_scales(coeffs, high_freq_levels)
mf_volatility = reconstruct_signal_at_scales(coeffs, medium_freq_levels)
lf_volatility = reconstruct_signal_at_scales(coeffs, low_freq_levels)
min_len = len(P_actual_kw)
df_master = pd.DataFrame({
    'P_clearsky_kw': P_clearsky_kw.values,
    'Volatility_High_Freq': hf_volatility[:min_len],
    'Volatility_Medium_Freq': mf_volatility[:min_len],
    'Volatility_Low_Freq': lf_volatility[:min_len]
}, index=P_actual_kw.index)

# --- 4. 核心计算：为每个分量计算波动强度 I_F ---
print(f"--- 步骤4：以 {TIME_RESOLUTION} 窗口计算各分量的波动强度 ---")
def calculate_fluctuation_intensity(df: pd.DataFrame, volatility_col: str, resolution: str):
    sigma_cloud = df[volatility_col].resample(resolution).std()
    # 【已修正】确保 px_clearsky 基于正确的列进行计算
    px_clearsky = df['P_clearsky_kw'].resample(resolution).mean() 
    df_analysis = pd.DataFrame({'Px_clearsky': px_clearsky, 'sigma_cloud': sigma_cloud})
    df_analysis.dropna(inplace=True)
    df_analysis = df_analysis[df_analysis['Px_clearsky'] > 0.01 * P_RATED_KW]
    df_analysis['I_F_PV'] = df_analysis['sigma_cloud'] / df_analysis['Px_clearsky']
    return df_analysis

components = {'High_Freq': 'Volatility_High_Freq', 'Medium_Freq': 'Volatility_Medium_Freq', 'Low_Freq': 'Volatility_Low_Freq'}
results = {name: calculate_fluctuation_intensity(df_master, col, TIME_RESOLUTION) for name, col in components.items()}

# --- 5. 可视化与数据保存 ---
print("--- 步骤5：生成组合分析图表并保存绘图数据 ---")
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
plot_titles = {'High_Freq': '高频分量 I_F 分布', 'Medium_Freq': '中频分量 I_F 分布', 'Low_Freq': '低频分量 I_F vs P_st'}

# --- 5a. 高频分量：分布分析与数据保存 ---
ax_hf = axes[0]
data_hf = results['High_Freq']['I_F_PV'].dropna()
if not data_hf.empty:
    df_dist_hf = pd.DataFrame({'I_F_PV_High_Freq': data_hf})
    output_dist_hf_path = OUT / f"distribution_data_High_Freq_{TIME_RESOLUTION}.csv"
    df_dist_hf.to_csv(output_dist_hf_path, index=False)
    print(f"  - 已将高频分量分布数据保存至: {output_dist_hf_path}")
    sns.histplot(data_hf, kde=True, stat="density", bins=50, ax=ax_hf, label='数据分布 (KDE)')
    try:
        log_data = np.log(data_hf[data_hf > 0])
        mu, sigma = stats.norm.fit(log_data)
        s_fit, scale_fit = sigma, np.exp(mu)
        xmin, xmax = ax_hf.get_xlim()
        x_pdf = np.linspace(xmin, xmax, 200)
        pdf = stats.lognorm.pdf(x_pdf, s=s_fit, scale=scale_fit)
        ax_hf.plot(x_pdf, pdf, 'r-', lw=2.5, label=f'对数正态拟合\n(s={s_fit:.2f}, scale={scale_fit:.2f})')
    except (ValueError, RuntimeError):
        print(" - 高频分量分布拟合失败。")
ax_hf.set_title(plot_titles['High_Freq'], fontsize=16)
ax_hf.set_xlabel('波动强度 I_F')
ax_hf.set_ylabel('概率密度')
ax_hf.legend()
ax_hf.grid(True, linestyle='--')

# --- 5b. 中频分量：分布分析与数据保存 ---
ax_mf = axes[1]
data_mf = results['Medium_Freq']['I_F_PV'].dropna()
if not data_mf.empty:
    df_dist_mf = pd.DataFrame({'I_F_PV_Medium_Freq': data_mf})
    output_dist_mf_path = OUT / f"distribution_data_Medium_Freq_{TIME_RESOLUTION}.csv"
    df_dist_mf.to_csv(output_dist_mf_path, index=False)
    print(f"  - 已将中频分量分布数据保存至: {output_dist_mf_path}")
    sns.histplot(data_mf, kde=True, stat="density", bins=50, ax=ax_mf, label='数据分布 (KDE)')
    try:
        log_data = np.log(data_mf[data_mf > 0])
        mu, sigma = stats.norm.fit(log_data)
        s_fit, scale_fit = sigma, np.exp(mu)
        xmin, xmax = ax_mf.get_xlim()
        x_pdf = np.linspace(xmin, xmax, 200)
        pdf = stats.lognorm.pdf(x_pdf, s=s_fit, scale=scale_fit)
        ax_mf.plot(x_pdf, pdf, 'r-', lw=2.5, label=f'对数正态拟合\n(s={s_fit:.2f}, scale={scale_fit:.2f})')
    except (ValueError, RuntimeError):
        print(" - 中频分量分布拟合失败。")
ax_mf.set_title(plot_titles['Medium_Freq'], fontsize=16)
ax_mf.set_xlabel('波动强度 I_F')
ax_mf.set_ylabel('')
ax_mf.legend()
ax_mf.grid(True, linestyle='--')

# --- 5c. 低频分量：回归分析与数据保存 ---
ax_lf = axes[2]
df_plot_lf = results['Low_Freq']
if not df_plot_lf.empty:
    x_data = df_plot_lf['Px_clearsky'] / P_RATED_KW
    y_data = df_plot_lf['I_F_PV']
    valid_indices = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0)
    x_fit, y_fit = x_data[valid_indices], y_data[valid_indices]
    df_scatter_lf = pd.DataFrame({'P_st_pu': x_fit, 'I_F_PV': y_fit})
    output_scatter_lf_path = OUT / f"scatter_data_Low_Freq_{TIME_RESOLUTION}.csv"
    df_scatter_lf.to_csv(output_scatter_lf_path, index=False)
    print(f"  - 已将低频分量散点数据保存至: {output_scatter_lf_path}")
    ax_lf.scatter(x_fit, y_fit, alpha=0.3, label=f'{TIME_RESOLUTION} 数据点')
    try:
        params, _ = curve_fit(power_law_model, x_fit, y_fit, p0=[0.1, -0.5, 0.1], maxfev=5000)
        a, beta, c = params
        x_curve = np.linspace(x_fit.min(), x_fit.max(), 100)
        y_curve = power_law_model(x_curve, a, beta, c)
        ax_lf.plot(x_curve, y_curve, color='red', linewidth=2.5, label=f'I={a:.2f}*P^{beta:.2f} + {c:.2f}')
    except (RuntimeError, ValueError):
        print("  - 低频分量的曲线拟合失败。")
ax_lf.set_title(plot_titles['Low_Freq'], fontsize=16)
ax_lf.set_xlabel('晴空平稳分量 P_st (p.u.)')
ax_lf.set_ylabel('')
ax_lf.legend()
ax_lf.grid(True, linestyle='--')
ax_lf.set_xlim(left=0)

# 【已清理】删除了脚本末尾多余、错误的代码

plt.suptitle(f'小波重构分量的混合分析 (时间窗口: {TIME_RESOLUTION})', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
output_path = OUT / "wavelet_hybrid_analysis_corrected_fit.png"
plt.savefig(output_path, dpi=150)
plt.show()
print(f"\n分析图表已成功保存至: {output_path}")