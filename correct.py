# -*- coding: utf-8 -*-
# 目标：手动修正 P_actual 数据中存在的时间戳滞后或超前问题

import pandas as pd
import matplotlib.pyplot as plt
import pvlib
from pathlib import Path

# ===================================================================
# 【重要】参数配置
# ===================================================================
# 1. 光伏电站与数据文件信息 (请确保与您之前的脚本一致)
LATITUDE = 34.05
LONGITUDE = -118.24
ALTITUDE = 71
TIMEZONE = "America/Los_Angeles"
INP = Path("outputs/clean")
OUT = Path("outputs/clean"); OUT.mkdir(parents=True, exist_ok=True) # 输出修正后的文件到同一目录
P_RATED_KW = 30.1

# 2. --- !!! 请根据“步骤1”的诊断图，在这里填入您观察到的偏移量 !!! ---
# 正数: P_actual 滞后了 (例如, 峰值晚了2分钟，就填 2)
# 负数: P_actual 超前了 (例如, 峰值早了3分钟，就填 -3)
LAG_TO_CORRECT_MINUTES = 30 # <--- 请修改这个数值

# 3. --- 选择一个晴朗的日子用于诊断和验证 ---
#    (您可以多试几个日期，找到云最少、曲线最平滑的一天)
CLEAR_DAY_TO_INSPECT = '2019-09-19' 
# ===================================================================


# --- 数据加载与晴空模型计算 ---
print("--- 正在加载数据并计算晴空模型 ---")
def read_csv_tz(path_csv: Path, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv, index_col=0)
    idx = pd.to_datetime(df.index, utc=True).tz_convert(tz)
    df.index = idx
    return df

pv_1m = read_csv_tz(INP/"pv_1min_clean.csv", TIMEZONE)
pcol = next((c for c in pv_1m.columns if c.lower() in ["p_kw","power","power_kw"]), pv_1m.columns[0])
P_actual_kw = pd.to_numeric(pv_1m[pcol], errors="coerce").fillna(0.0).clip(lower=0.0)

location = pvlib.location.Location(latitude=LATITUDE, longitude=LONGITUDE, altitude=ALTITUDE, tz=TIMEZONE)
clearsky_ghi = location.get_clearsky(P_actual_kw.index)['ghi']
# 这里我们只需要一个粗略的晴空曲线用于对齐，无需复杂的标定
scaling_factor = P_actual_kw.max() / clearsky_ghi.max()
P_clearsky_kw = clearsky_ghi * scaling_factor


# --- 步骤 1: 诊断滞后量 (可视化修正前的数据) ---
print(f"\n--- 步骤1：诊断 {CLEAR_DAY_TO_INSPECT} 的时间偏移 ---")
plt.figure(figsize=(12, 7))
plt.plot(P_actual_kw.loc[CLEAR_DAY_TO_INSPECT], label='实际功率 (修正前)', color='blue')
plt.plot(P_clearsky_kw.loc[CLEAR_DAY_TO_INSPECT], label='晴空功率 (时间基准)', color='red', linestyle='--')
plt.title(f'修正前功率曲线对比 ({CLEAR_DAY_TO_INSPECT})\n请放大观察峰值时刻，估算时间差', fontsize=16)
plt.xlabel('时间')
plt.ylabel('功率 (kW)')
plt.legend()
plt.grid(True)
plt.show()

# --- 步骤 2: 执行手动修正 ---
print(f"\n--- 步骤2：执行修正。将实际功率的时间戳向前平移 {-LAG_TO_CORRECT_MINUTES} 分钟 ---")
# 创建一个新的Series用于存储修正后的数据，避免修改原始数据
P_actual_kw_corrected = P_actual_kw.copy()

# 使用 pandas.Timedelta 来平移时间索引
# 如果滞后了2分钟 (LAG_TO_CORRECT_MINUTES = 2), 我们需要将时间戳减去2分钟，让它“赶上来”
time_shift = pd.to_timedelta(LAG_TO_CORRECT_MINUTES, unit='m')
P_actual_kw_corrected.index = P_actual_kw_corrected.index - time_shift

print("时间戳修正完成。")

# --- 步骤 3: 验证修正效果 (可视化修正后的数据) ---
print(f"\n--- 步骤3：验证 {CLEAR_DAY_TO_INSPECT} 的修正效果 ---")
plt.figure(figsize=(12, 7))
plt.plot(P_actual_kw_corrected.loc[CLEAR_DAY_TO_INSPECT], label='实际功率 (修正后)', color='green')
plt.plot(P_clearsky_kw.loc[CLEAR_DAY_TO_INSPECT], label='晴空功率 (时间基准)', color='red', linestyle='--')
plt.title(f'修正后功率曲线对比 ({CLEAR_DAY_TO_INSPECT})\n检查峰值是否对齐', fontsize=16)
plt.xlabel('时间')
plt.ylabel('功率 (kW)')
plt.legend()
plt.grid(True)
plt.show()

# --- 步骤 4: 保存修正后的数据 ---
# 创建一个新的DataFrame来保存修正后的功率数据
corrected_df = pd.DataFrame(P_actual_kw_corrected)
corrected_df.columns = [pcol] # 保持原始的列名

# 定义新的文件名并保存
output_path = OUT / "pv_1min_clean_corrected.csv"
corrected_df.to_csv(output_path)
print(f"\n--- 步骤4：修正后的数据已保存 ---")
print(f"新的数据文件位于: {output_path}")
print("请在您后续的所有分析脚本中，将输入文件路径 INP 指向这个新的 'corrected' 文件。")