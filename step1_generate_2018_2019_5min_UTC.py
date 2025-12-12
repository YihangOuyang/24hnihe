# -*- coding: utf-8 -*-
"""
统一 2018–2019 年 Stanford 实测功率 & NSRDB 气象数据 (High-Quality Preprocessing)：
- 全部时间统一为 UTC、5 分钟分辨率
- 对光伏功率做夜间清零
- [新增] 死值检测与物理填充 (Physics-guided Imputation)
- 最终输出 merged_2018_2019_5min_UTC_cleaned.csv
"""

import os
import pandas as pd
import numpy as np
import pvlib  # [新增] 用于光伏晴空模型计算
from sklearn.linear_model import LinearRegression # [新增] 用于拟合转换效率

# ===================== 用户配置 =====================

BASE_DIR = r'./rawdata'       # 改成你保存 csv 的目录
PV_FILES = {
    2018: '2018_pv_raw.csv',
    2019: '2019_pv_raw.csv',
}
NSRDB_FILES = {
    2018: 'standford_2018_5min_localtime.csv',
    2019: 'standford_2019_5min_localtime.csv',
}

POWER_COLUMN = 'Huang_E4102_kW'  # 实测功率列名
POWER_THRESHOLD = 0.0            # 夜间清零阈值（< 阈值的全部置 0）
PV_LOCAL_TZ = 'US/Pacific'       # Stanford 本地时区（含夏令时）
OUTPUT_FILE = os.path.join(BASE_DIR, 'merged_2018_2019_5min_UTC_cleaned.csv')

# [新增] 死值检测与填充配置
STUCK_WINDOW = 12                # 连续多少个点(5min*12=1h)值不变视为死值
STUCK_STD_THRESHOLD = 1e-4       # 标准差低于此值视为不变
MIN_GHI_FOR_FIT = 50             # 拟合时只用 GHI > 50 的数据


# ===================== NSRDB 处理函数 =====================

def get_nsrdb_meta(path: str):
    """
    [修改] 读取 NSRDB CSV 第一行，获取 Time Zone 以及经纬度信息。
    """
    meta = pd.read_csv(path, nrows=1)
    tz_offset = meta['Time Zone'].iloc[0]
    
    # 提取地理信息供 pvlib 使用
    lat = meta['Latitude'].iloc[0]
    lon = meta['Longitude'].iloc[0]
    # NSRDB 有时表头是 Elevation 有时是 Altitude，这里做个兼容，如果没有默认0
    alt = meta.get('Elevation', meta.get('Altitude', pd.Series([0]))).iloc[0]

    if tz_offset < 0:
        tz_str = f"Etc/GMT+{abs(int(tz_offset))}"
    elif tz_offset > 0:
        tz_str = f"Etc/GMT-{int(tz_offset)}"
    else:
        tz_str = "UTC"

    return tz_str, lat, lon, alt


def load_nsrdb_year(filename: str):
    """
    读取一年的 NSRDB 5min 数据：
    - [修改] 返回 DataFrame 以及地理位置信息 (lat, lon, alt)
    """
    path = os.path.join(BASE_DIR, filename)
    print(f"[NSRDB] 加载气象数据：{path}")

    tz_str, lat, lon, alt = get_nsrdb_meta(path)
    print(f"[NSRDB] 检测到 Time Zone = {tz_str}, Location=({lat}, {lon})")

    # 第一行是 meta，第二行开始才是 Year, Month, Day, Hour, Minute ...
    df = pd.read_csv(path, skiprows=2)

    # 构造 datetime 索引（先不带时区）
    dt_index = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df.index = dt_index

    # 本地化为 NSRDB 定义的固定时区（无夏令时），再转为 UTC
    df = df.tz_localize(tz_str)
    df = df.tz_convert('UTC')

    col_map = {
        'GHI': 'ghi',
        'DHI': 'dhi',
        'DNI': 'dni',
        'Temperature': 'temp_air',
        'Wind Speed': 'wind_speed',
        'Relative Humidity': 'rh',
        # 有些 NSRDB 版本自带 Clearsky GHI，保留下来备用，没有也没关系
        'Clearsky GHI': 'ghi_cs_nsrdb'
    }
    used_cols = {src: dst for src, dst in col_map.items() if src in df.columns}

    df_sel = df[list(used_cols.keys())].rename(columns=used_cols)

    print(f"[NSRDB] 5min 行数：{len(df_sel)}")
    
    # 返回数据和位置元组
    return df_sel, (lat, lon, alt)


# ===================== 实测功率处理函数 =====================

def load_pv_year(year: int, filename: str) -> pd.DataFrame:
    """
    读取某一年的 Stanford 实测功率数据 (保持不变)
    """
    path = os.path.join(BASE_DIR, filename)
    print(f"[PV] 加载 {year} 年功率数据：{path}")

    df = pd.read_csv(path)

    df['timestamp_local'] = pd.to_datetime(df['Date'])
    df = df.set_index('timestamp_local')

    idx = df.index
    amb_flags = np.zeros(len(idx), dtype=bool)
    first_dup = idx.duplicated(keep='first')
    amb_flags[first_dup] = True

    df = df.tz_localize(
        PV_LOCAL_TZ,
        ambiguous=amb_flags,       
        nonexistent='shift_forward'
    )
    df = df.tz_convert('UTC')

    df[POWER_COLUMN] = df[POWER_COLUMN].where(
        df[POWER_COLUMN] >= POWER_THRESHOLD,
        0.0
    )

    df_5min = df.resample('5T').mean(numeric_only=True)
    df_5min = df_5min.rename(columns={POWER_COLUMN: 'power_kw'})
    df_5min['year'] = year

    print(f"[PV] {year} 年 5min 行数：{len(df_5min)} "
          f"(UTC 索引是否有重复？ {df_5min.index.duplicated().sum() == 0})")
    return df_5min[['power_kw', 'year']]


# ===================== [新增] 数据清洗与填充核心算法 =====================

def clean_and_impute_stuck_data(df: pd.DataFrame, lat, lon, alt):
    """
    Top-tier Journal Level Preprocessing (v4 - Final Logic):
    1. Detect stuck values (Non-zero stuck AND Daytime zero stuck).
    2. Physics-guided Imputation (ClearSky model).
    3. Robust Relative Noise (5%).
    """
    print("\n[QC] 开始执行数据清洗与填充程序 (v4 - 包含白天0值修复)...")
    
    # =======================================================
    # 0. 预先计算晴空辐射 (GHI CS) - 这一步必须提到最前面
    #    因为检测"白天0值"需要用到它
    # =======================================================
    print("[QC] 计算 pvlib 晴空辐射模型 (Ineichen)...")
    site_location = pvlib.location.Location(lat, lon, altitude=alt, name='Stanford')
    cs = site_location.get_clearsky(df.index) 
    df['ghi_cs_calc'] = cs['ghi']

    # =======================================================
    # 1. 标记“死值” (Stuck Values) - 逻辑升级
    # =======================================================
    rolling_std = df['power_kw'].rolling(window=STUCK_WINDOW, center=True).std()
    
    # 情况 A: 非零死值 (传感器卡死在某数值)
    # 逻辑: 波动极小，且数值 > 0.1
    mask_stuck_nonzero = (rolling_std < STUCK_STD_THRESHOLD) & (df['power_kw'] > 0.1)
    
    # 情况 B: 白天零值 (断路/停机)
    # 逻辑: 波动极小(就是0)，数值 < 0.1，但是理论晴空辐射 > 10 (说明此时必须有光)
    mask_stuck_zero_daytime = (rolling_std < STUCK_STD_THRESHOLD) & \
                              (df['power_kw'] <= 0.001) & \
                              (df['ghi_cs_calc'] > 10)
    
    # 合并两种异常情况
    is_stuck = mask_stuck_nonzero | mask_stuck_zero_daytime
    
    # 膨胀标记范围 (填补窗口边缘)
    is_stuck = is_stuck.rolling(window=3, center=True).max().astype(bool).fillna(False)
    
    # 强制第一个点不处理
    if len(is_stuck) > 0:
        is_stuck.iloc[0] = False

    bad_count = is_stuck.sum()
    print(f"[QC] 检测到异常死值数据点：{bad_count} 个 (含白天异常0值)")
    
    if bad_count == 0:
        return df

    # =======================================================
    # 2. 拟合转换系数 (Efficiency Fitting)
    # =======================================================
    # 仅使用正常数据拟合 k 值
    valid_mask = (~is_stuck) & (df['ghi'] > 50) & (df['power_kw'] > 0)
    
    if valid_mask.sum() > 100:
        X = df.loc[valid_mask, ['ghi']] 
        y = df.loc[valid_mask, 'power_kw']
        
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        coef = reg.coef_[0]
        print(f"[QC] 拟合转换效率 k={coef:.4f}")
    else:
        coef = df['power_kw'].max() / df['ghi'].max() if df['ghi'].max() > 1 else 0
        print("[QC] 警告：有效数据不足，使用粗略比率。")

    # =======================================================
    # 3. 生成填充值 (Relative Noise)
    # =======================================================
    
    # 3.1 基础值
    base_power = df.loc[is_stuck, 'ghi_cs_calc'] * coef
    
    # 3.2 动态相对噪声 (5%)
    noise_level = 0.05 
    random_factors = np.random.normal(loc=0, scale=1.0, size=len(base_power))
    imputed_values = base_power + (base_power * noise_level * random_factors)
    
    # 3.3 物理约束
    # 夜间强制归零 (防止在日出日落边缘填充出噪音)
    night_mask = df.loc[is_stuck, 'ghi_cs_calc'] < 1.0
    imputed_values[night_mask] = 0.0
    
    # 修正负数
    imputed_values = imputed_values.clip(lower=0)
    
    # =======================================================
    # 4. 应用填充
    # =======================================================
    df.loc[is_stuck, 'power_kw'] = imputed_values
    
    df['is_imputed'] = 0
    df.loc[is_stuck, 'is_imputed'] = 1
    
    print("[QC] 异常数据填充完成。")
    return df


# ===================== 主流程 =====================

def main():
    # 1) 2018/2019 实测功率
    pv_list = []
    for y, fname in sorted(PV_FILES.items()):
        pv_list.append(load_pv_year(y, fname))
    pv_all = pd.concat(pv_list).sort_index()
    print(f"[PV] 合并后总行数：{len(pv_all)}")

    # 2) 2018/2019 NSRDB
    w_list = []
    # 用来临时存储地理位置（假设同个电站位置不变，取最后一个文件的位置即可）
    final_lat, final_lon, final_alt = 0, 0, 0
    
    for y, fname in sorted(NSRDB_FILES.items()):
        # [修改] 这里接收两个返回值
        df_w, (lat, lon, alt) = load_nsrdb_year(fname)
        w_list.append(df_w)
        final_lat, final_lon, final_alt = lat, lon, alt
        
    weather_all = pd.concat(w_list).sort_index()
    print(f"[NSRDB] 合并后总行数：{len(weather_all)}")

    # 3) UTC 时间轴上合并
    merged = weather_all.join(pv_all, how='inner')
    merged.index.name = 'timestamp_utc'

    print(f"[MERGE] 合并后行数：{len(merged)}")
    print(f"[MERGE] 列：{merged.columns.tolist()}")

    # 4) [新增] 执行死值检测与填充
    # 传入刚才获取的经纬度
    merged = clean_and_impute_stuck_data(merged, final_lat, final_lon, final_alt)

    # 5) 导出
    # 移除计算过程中的临时列，保持输出整洁，但保留 'is_imputed' 供你参考
    if 'ghi_cs_calc' in merged.columns:
        merged = merged.drop(columns=['ghi_cs_calc'])
        
    merged.to_csv(OUTPUT_FILE)
    print(f"[DONE] 文件已保存：{OUTPUT_FILE}")
    print("[INFO] 'is_imputed' 列为 1 表示该点为填充数据。")


if __name__ == "__main__":
    main()