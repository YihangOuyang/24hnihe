# -*- coding: utf-8 -*-
"""
专门用于检查物理时间对齐的脚本
Check if P_actual and P_clearsky peaks are aligned.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ================= 配置区域 =================
# 请确保这些参数与 train_main.py 完全一致
CURRENT_DIR = Path(__file__).parent.absolute()
DATA_FILE = CURRENT_DIR / "rawdata" / "Final_Training_Dataset_15min.csv"
OUTPUT_IMG = CURRENT_DIR / "outputs" / "alignment_check.png"

# 你的站点参数
LATITUDE = 37.427963
LONGITUDE = -122.154785
CAPACITY = 30.1 
TARGET_COL = 'power_kw' # 你的实际功率列名

# [关键] 待验证的时间位移
PHYSICS_TIME_SHIFT_MINUTES = 0
# ===========================================

def check_alignment():
    print(f"--- 正在检查时间对齐 (Shift = {PHYSICS_TIME_SHIFT_MINUTES} min) ---")
    
    if not DATA_FILE.exists():
        print(f"❌ 数据文件未找到: {DATA_FILE}")
        return

    # 1. 加载数据
    df = pd.read_csv(DATA_FILE, index_col=0)
    # 解析时间并转为当地时区 (确保与训练逻辑一致)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('Etc/GMT+8')
    
    # 2. 计算物理基准 (P_clearsky)
    # 创建物理时间轴
    times = df.index
    times_phys = times + pd.Timedelta(minutes=PHYSICS_TIME_SHIFT_MINUTES)
    
    loc = pvlib.location.Location(LATITUDE, LONGITUDE, tz=times.tz)
    
    # 计算晴空辐射
    print("   计算 P_clearsky ...")
    cs = loc.get_clearsky(times_phys)
    
    # 估算晴空功率 (简化公式，与训练一致)
    # P_clear ≈ GHI * Capacity / 1000
    p_clearsky = cs['ghi'] * (CAPACITY / 1000.0)
    
    df['P_actual'] = df[TARGET_COL]
    df['P_clearsky'] = p_clearsky.values
    
    # 3. 自动寻找一个完美的晴天 (用于画图)
    # 逻辑：找实际功率峰值很高，且接近 P_clearsky 的一天
    daily_max = df.groupby(df.index.date)['P_actual'].max()
    # 找前 5 个发电量最大的日子
    top_days = daily_max.nlargest(5).index
    
    print(f"   选取典型晴天进行绘图: {top_days[0]}")
    
    # 4. 绘图验证
    # 我们画 2 天的数据，以便看清日出日落
    target_day = top_days[0]
    start_plot = pd.Timestamp(target_day).tz_localize(df.index.tz)
    end_plot = start_plot + pd.Timedelta(days=2) # 画2天
    
    subset = df[(df.index >= start_plot) & (df.index < end_plot)]
    
    plt.figure(figsize=(12, 6))
    
    # 画实际功率 (黑色实线)
    plt.plot(subset.index, subset['P_actual'], color='black', linewidth=2, label='Actual Power (Ground Truth)')
    
    # 画理论晴空功率 (红色虚线)
    plt.plot(subset.index, subset['P_clearsky'], color='red', linestyle='--', linewidth=2, label=f'P_clearsky (Shift={PHYSICS_TIME_SHIFT_MINUTES}m)')
    
    plt.title(f"Time Alignment Check: {target_day} (Shift={PHYSICS_TIME_SHIFT_MINUTES}min)", fontsize=14)
    plt.ylabel("Power (kW)")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 格式化 X 轴
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m-%d'))
    
    # 确保输出目录存在
    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_IMG)
    print(f"✅ 检查图已保存至: {OUTPUT_IMG}")
    print("   请打开图片检查：红线和黑线的峰值是否重合？")
    print("   - 如果红线在黑线 左边：说明 Shift 太负了 (或者是正的)")
    print("   - 如果红线在黑线 右边：说明 Shift 不够负")

if __name__ == "__main__":
    check_alignment()