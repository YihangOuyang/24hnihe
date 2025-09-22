# -*- coding: utf-8 -*-
# 思路：模仿PDF中的风电研究方法，为光伏建立“波动强度”与“平稳分量”的解析表达式

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib
from pathlib import Path
from scipy.optimize import curve_fit

# ===================================================================
# 【重要】需要您根据实际情况修改的参数
# ===================================================================
# 1. 光伏电站的地理位置信息
LATITUDE = 34.05
LONGITUDE = -118.24
ALTITUDE = 71
TIMEZONE = "America/Los_Angeles"

# 2. 您的数据文件路径和额定功率
INP = Path("outputs/clean")
OUT = Path("outputs/pv_power_law_analysis"); OUT.mkdir(parents=True, exist_ok=True)
P_RATED_KW = 30.1
# ===================================================================

# --- 1. 读取实际功率数据 (定义光伏的“实际功率”) ---
def read_csv_tz(path_csv: Path, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv, index_col=0)
    idx = pd.to_datetime(df.index, utc=True).tz_convert(tz)
    df.index = idx
    return df

pv_1m  = read_csv_tz(INP/"pv_1min_clean.csv", TIMEZONE)
pcol = next((c for c in pv_1m.columns if c.lower() in ["p_kw","power","power_kw"]), pv_1m.columns[0])
P_actual_kw = pd.to_numeric(pv_1m[pcol], errors="coerce").fillna(0.0).clip(lower=0.0)
print("步骤1：已加载实际光伏功率数据。")

# --- 2. 生成晴空功率 (定义光伏的“平稳分量”) ---
location = pvlib.location.Location(latitude=LATITUDE, longitude=LONGITUDE, altitude=ALTITUDE, tz=TIMEZONE)
clearsky = location.get_clearsky(P_actual_kw.index)
peak_actual_power = P_actual_kw.max()
peak_ghi_clearsky = clearsky['ghi'].max()
scaling_factor = peak_actual_power / peak_ghi_clearsky if peak_ghi_clearsky > 0 else 0
P_clearsky_kw = (clearsky['ghi'] * scaling_factor).clip(lower=0)
print(f"步骤2：已生成晴空功率作为“平稳分量”。缩放系数为 {scaling_factor:.4f}。")

# --- 3. 计算“云瞬变” (定义光伏的“波动分量”) ---
cloud_transient = (P_actual_kw - P_clearsky_kw)
cloud_transient[P_clearsky_kw < 0.001 * P_RATED_KW] = 0
print("步骤3：已计算“云瞬变”作为“波动分量”。")

# --- 4. 计算光伏的“波动强度 I_F_PV” ---
# 严格遵循PDF中的定义 [cite: 213]
# a) 计算波动分量的标准差 σ_cloud
sigma_cloud_1h = cloud_transient.resample("1H").std().dropna()

# b) 计算平稳分量的平均值 Px_clearsky
Px_clearsky_1h = P_clearsky_kw.resample("1H").mean().dropna()

# c) 计算波动强度 I_F = σ / Px
df_analysis = pd.DataFrame({
    'Px_clearsky': Px_clearsky_1h,
    'sigma_cloud': sigma_cloud_1h
}).dropna()
df_analysis = df_analysis[df_analysis['Px_clearsky'] > 0.01 * P_RATED_KW] # 剔除功率过低的点
df_analysis['I_F_PV'] = df_analysis['sigma_cloud'] / df_analysis['Px_clearsky']
print("步骤4：已计算出每个小时的波动强度 I_F_PV。")

# --- 5. 拟合解析表达式 (三参数幂律模型) ---
# 定义与PDF中形式相同的幂律模型函数 
def power_law_model(p, a, beta, c):
    return a * np.power(p, beta) + c

# 准备拟合数据 (转换为标幺值 p.u. 以便与PDF中的图对比)
x_data = df_analysis['Px_clearsky'] / P_RATED_KW
y_data = df_analysis['I_F_PV']

# 过滤掉无效数据
valid_indices = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0)
x_fit = x_data[valid_indices]
y_fit = y_data[valid_indices]

try:
    # 使用scipy.optimize.curve_fit进行最小二乘拟合
    params, covariance = curve_fit(power_law_model, x_fit, y_fit, p0=[0.1, -0.5, 0.1], maxfev=5000)
    a, beta, c = params
    print("\n步骤5：成功拟合三参数幂律模型！")
    print("="*40)
    print("      光伏波动强度解析表达式:")
    print(f"      I_F = {a:.4f} * P_st^{beta:.4f} + {c:.4f}")
    print("="*40)

    # --- 6. 可视化拟合结果 ---
    plt.figure(figsize=(10, 7))
    plt.scatter(x_fit, y_fit, alpha=0.3, label='实际数据点')
    
    # 生成拟合曲线的X值
    x_curve = np.linspace(x_fit.min(), x_fit.max(), 200)
    y_curve = power_law_model(x_curve, a, beta, c)
    
    plt.plot(x_curve, y_curve, color='red', linewidth=2.5, label='幂律模型拟合曲线')
    
    plt.title('光伏波动强度 vs. 晴空平稳分量 (模仿PDF方法)', fontsize=16)
    plt.xlabel('晴空平稳分量 P_st (p.u.)', fontsize=12)
    plt.ylabel('光伏波动强度 I_F', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=0)
    plt.ylim(-1,5)
    plt.savefig(OUT/"pv_fluctuation_intensity_fit.png", dpi=150)
    plt.show()

except RuntimeError:
    print("\n[ERROR] 曲线拟合失败。可能是数据量太少或数据分布不适合幂律模型。")