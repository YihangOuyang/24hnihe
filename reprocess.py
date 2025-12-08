# A_save_clean_data.py
import pandas as pd, numpy as np, re
from pathlib import Path

# ==== 关键修改区域 ====
# 1. 输入数据的时区（Excel 通常记录的是当地墙上时间，含夏令时）
INPUT_TZ = "America/Los_Angeles"
# 2. 输出数据的时区（为了训练稳定，强制统一为标准时，与 NWP 一致）
TARGET_TZ = "Etc/GMT+8"

PV_XLSX = "2019_pv_raw.xlsx"
WEA_CSV = "merged_pv_and_weather_data.csv"
OUTDIR = Path("outputs/clean"); OUTDIR.mkdir(parents=True, exist_ok=True)

# ... (to_float_series 函数保持不变) ...
def to_float_series(s: pd.Series) -> pd.Series:
    x = s.astype(str)
    x = (x.str.replace('\u2212','-',regex=False)
           .str.replace('\u2013','-',regex=False)
           .str.replace('\u2014','-',regex=False)
           .str.replace(',','',regex=False)
           .str.replace('\u00A0','',regex=False)
           .str.strip())
    x = x.str.replace(r"[^0-9eE+\-\.]", "", regex=True)
    return pd.to_numeric(x, errors="coerce")

# 1) 读取 1-min PV
print("Processing PV Data...")
pv_raw = pd.read_excel(PV_XLSX)
pv_raw.columns = pv_raw.columns.str.strip()
tcol = [c for c in pv_raw.columns if c.lower().startswith(("timest","timestamp"))][0]
pcol = [c for c in pv_raw.columns if "power" in c.lower()][0]

# 解析时间
t = pd.to_datetime(pv_raw[tcol].astype(str).str.strip(),
                   format="%Y-%m-%dT%H:%M:%S", errors="coerce")

# 【核心逻辑修正】
# 第一步：告诉 Pandas 这是洛杉矶当地时间（处理夏令时歧义）
t = t.dt.tz_localize(INPUT_TZ, nonexistent="shift_forward", ambiguous="NaT")
# 第二步：立刻转换为目标标准时区（与 NWP 对齐）
# 这一步会将 13:00(PDT) 变为 12:00(PST)，从而消除“左移”现象
t = t.dt.tz_convert(TARGET_TZ)

P_kW_series = to_float_series(pv_raw[pcol])
pv_1m = pd.Series(P_kW_series.to_numpy(), index=t, name="P_KW").sort_index()

# 重采样插值（注意：这里使用 TARGET_TZ）
pv_1m = pv_1m.reindex(pd.date_range(pv_1m.index.min(), pv_1m.index.max(),
                                    freq="1min", tz=TARGET_TZ)).interpolate("time")

# 保存（注意：date_format 中 %z 会显示 -0800，确保你看到的是标准时）
pv_1m.to_frame().to_csv(OUTDIR/"pv_1min_clean.csv",
                        date_format="%Y-%m-%dT%H:%M:%S%z")

# 2) 读取 5-min 气象
print("Processing Weather Data...")
wea = pd.read_csv(WEA_CSV)
wea.columns = [c.strip().lower() for c in wea.columns]
if "wind_spee" in wea.columns and "wind_speed" not in wea.columns:
    wea = wea.rename(columns={"wind_spee":"wind_speed"})
tcol = [c for c in wea.columns if c.startswith("timestamp")][0]
ti = pd.to_datetime(wea[tcol], errors="coerce")

# 气象数据时区处理
if ti.dt.tz is None:
    # 假设气象数据也是当地时间记录的
    ti = ti.dt.tz_localize(INPUT_TZ, ambiguous="NaT")
else:
    # 如果自带时区，先转到 INPUT 确认无误（可选），再转 TARGET
    ti = ti.dt.tz_convert(INPUT_TZ)

# 统一转为标准时
ti = ti.dt.tz_convert(TARGET_TZ)

wea = wea.set_index(ti).sort_index()
for c in ["ghi","dni","dhi","temp_air","wind_speed"]:
    if c not in wea.columns: continue
    wea[c] = to_float_series(wea[c])
if "albedo" not in wea.columns:
    wea["albedo"] = 0.2

# 筛选存在的列
cols_to_keep = [c for c in ["ghi","dni","dhi","temp_air","wind_speed","albedo"] if c in wea.columns]
wea_1m = wea[cols_to_keep].resample("1min").interpolate("time")

# 物理约束
for c in ["ghi","dni","dhi"]:
    if c in wea_1m.columns:
        wea_1m[c] = wea_1m[c].clip(lower=0)

wea_1m.to_csv(OUTDIR/"weather_1min_clean.csv", date_format="%Y-%m-%dT%H:%M:%S%z")

print(f"✅ 阶段A完成：数据已统一对齐到 {TARGET_TZ} (固定标准时)")