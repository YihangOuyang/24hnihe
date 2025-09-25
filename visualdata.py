# -*- coding: utf-8 -*-
# 目标：可视化几天的光伏功率曲线，并与小波重构的各分量进行对比

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib
from pathlib import Path
import pywt
import matplotlib.dates as mdates

# ===================================================================
# 【重要】参数配置
# ===================================================================
# 1. 光伏电站与数据文件信息 (请确保与您之前的脚本一致)
LATITUDE = 34.05
LONGITUDE = -118.24
ALTITUDE = 71
TIMEZONE = "America/Los_Angeles"
INP = Path("outputs/clean")
OUT = Path("outputs/pv_power_law_analysis"); OUT.mkdir(parents=True, exist_ok=True)
P_RATED_KW = 30.1

# 2. 【请在这里选择您想看的可视化日期范围】
START_DATE_TO_PLOT = '2019-09-18'
END_DATE_TO_PLOT = '2019-09-20'
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

# --- 2. 计算晴空功率与总波动分量 ---
print("--- 步骤2：计算晴空功率与总波动 ---")
location = pvlib.location.Location(latitude=LATITUDE, longitude=LONGITUDE, altitude=ALTITUDE, tz=TIMEZONE)
clearsky_ghi = location.get_clearsky(P_actual_kw.index)['ghi']
is_clear = pvlib.clearsky.detect_clearsky(P_actual_kw, clearsky_ghi, P_actual_kw.index)
clearsky_points_actual_power = P_actual_kw[is_clear]
clearsky_points_ghi = clearsky_ghi[is_clear]
if not clearsky_points_actual_power.empty:
    x = clearsky_points_ghi.values.reshape(-1, 1)
    y = clearsky_points_actual_power.values
    xtx = np.dot(x.T, x)
    xty = np.dot(x.T, y)
    scaling_factor = (xty / xtx)[0, 0] if xtx[0, 0] > 0 else 0
else:
    peak_actual_power = P_actual_kw.max()
    peak_ghi_clearsky = clearsky_ghi.max()
    scaling_factor = peak_actual_power / peak_ghi_clearsky if peak_ghi_clearsky > 0 else 0
P_clearsky_kw = (clearsky_ghi * scaling_factor).clip(lower=0)
cloud_transient_total = (P_actual_kw - P_clearsky_kw)
cloud_transient_total[P_clearsky_kw < 0.001 * P_RATED_KW] = 0

# --- 3. 小波分解与选择性重构 ---
print("--- 步骤3：重构不同时间尺度的波动分量 ---")
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
    'P_actual_kw': P_actual_kw.values,
    'P_clearsky_kw': P_clearsky_kw.values,
    'Volatility_High_Freq': hf_volatility[:min_len],
    'Volatility_Medium_Freq': mf_volatility[:min_len],
    'Volatility_Low_Freq': lf_volatility[:min_len]
}, index=P_actual_kw.index)

# --- 4. 准备绘图数据 ---
print("--- 步骤4：准备用于可视化的数据 ---")
# 计算每个分量重构后的总功率
df_master['Total_Power_LF'] = (df_master['P_clearsky_kw'] + df_master['Volatility_Low_Freq']).clip(lower=0)
df_master['Total_Power_MF'] = (df_master['P_clearsky_kw'] + df_master['Volatility_Medium_Freq']).clip(lower=0)
df_master['Total_Power_HF'] = (df_master['P_clearsky_kw'] + df_master['Volatility_High_Freq']).clip(lower=0)

# 根据您选择的日期范围筛选数据
df_plot = df_master.loc[START_DATE_TO_PLOT:END_DATE_TO_PLOT].copy()

# --- 5. 执行可视化 ---
print(f"--- 步骤5：正在绘制 {START_DATE_TO_PLOT} 到 {END_DATE_TO_PLOT} 的功率曲线 ---")
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
fig.suptitle(f'光伏功率与小波重构分量对比 ({START_DATE_TO_PLOT} to {END_DATE_TO_PLOT})', fontsize=18)

# 子图1: 实际功率 vs 晴空功率 (宏观对比)
axes[0].plot(df_plot.index, df_plot['P_actual_kw'], label='实际功率', color='dodgerblue', linewidth=2)
axes[0].plot(df_plot.index, df_plot['P_clearsky_kw'], label='晴空功率 (天花板)', color='red', linestyle='--', linewidth=2)
axes[0].set_title('整体功率对比', fontsize=14)
axes[0].legend()

# 子图2: 低频分量 (趋势)
axes[1].plot(df_plot.index, df_plot['P_actual_kw'], label='实际功率 (背景)', color='silver')
axes[1].plot(df_plot.index, df_plot['Total_Power_LF'], label='重构功率 (仅含低频)', color='green', linewidth=1.5)
axes[1].set_title('低频分量 (小时级趋势/计划)', fontsize=14)
axes[1].legend()

# 子图3: 中频分量 (主要波动)
axes[2].plot(df_plot.index, df_plot['P_actual_kw'], label='实际功率 (背景)', color='silver')
axes[2].plot(df_plot.index, df_plot['Total_Power_MF'], label='重构功率 (仅含中频)', color='orange', linewidth=1.5)
axes[2].set_title('中频分量 (分钟级波动/调度)', fontsize=14)
axes[2].legend()

# 子图4: 高频分量 (噪声/毛刺)
axes[3].plot(df_plot.index, df_plot['P_actual_kw'], label='实际功率 (背景)', color='silver')
axes[3].plot(df_plot.index, df_plot['Total_Power_HF'], label='重构功率 (仅含高频)', color='purple', linewidth=1.5)
axes[3].set_title('高频分量 (秒级-分钟级毛刺/调频)', fontsize=14)
axes[3].legend()

# 统一格式化
for ax in axes:
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylabel('功率 (kW)')
    ax.set_ylim(bottom=-1) # 留出一点下边距

# 优化X轴日期显示
ax.xaxis.set_major_locator(mdates.DayLocator()) # 主刻度为天
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18])) # 次刻度为6,12,18点
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%Hh'))
plt.xlabel("日期与时间")

# 自动调整布局并保存
plt.show()