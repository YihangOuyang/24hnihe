# A_save_clean_data.py
# 功能：修复奇怪负号/千分位，标准化时区，插值到 1min，并仅保存“清洗后的原始数据”
import pandas as pd, numpy as np, re
from pathlib import Path

# ==== 基本参数 ====
TZ = "America/Los_Angeles"
PV_XLSX = "2019_pv_raw.xlsx"                  # 1-min：列形如 Timestemp, Power (kW)
WEA_CSV = "merged_pv_and_weather_data.csv"    # 5-min：Timestamp_Local_UTC-08:00, ghi,dni,dhi,temp_air,wind_spee,albedo
OUTDIR = Path("outputs/clean"); OUTDIR.mkdir(parents=True, exist_ok=True)

# 数值规范化：修复 U+2212 等“非 ASCII 负号”、千分位逗号、不可见空白
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

# 1) 读取 1-min PV（仅清洗与补齐到 1min；不翻转、不裁负）
pv_raw = pd.read_excel(PV_XLSX)
pv_raw.columns = pv_raw.columns.str.strip()
tcol = [c for c in pv_raw.columns if c.lower().startswith(("timest","timestamp"))][0]
pcol = [c for c in pv_raw.columns if "power" in c.lower()][0]
t = pd.to_datetime(pv_raw[tcol].astype(str).str.strip(),
                   format="%Y-%m-%dT%H:%M:%S", errors="coerce")
t = t.dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")  # 本地化，不改变时刻
P_kW_series = to_float_series(pv_raw[pcol])
pv_1m = pd.Series(P_kW_series.to_numpy(), index=t, name="P_KW").sort_index()
pv_1m = pv_1m.reindex(pd.date_range(pv_1m.index.min(), pv_1m.index.max(),
                                    freq="1min", tz=TZ)).interpolate("time")
pv_1m.to_frame().to_csv(OUTDIR/"pv_1min_clean.csv",
                        date_format="%Y-%m-%dT%H:%M:%S%z")  # 保存含时区偏移

# 2) 读取 5-min 气象 → 1-min（仅清洗与插值）
wea = pd.read_csv(WEA_CSV)
wea.columns = [c.strip().lower() for c in wea.columns]
if "wind_spee" in wea.columns and "wind_speed" not in wea.columns:
    wea = wea.rename(columns={"wind_spee":"wind_speed"})
tcol = [c for c in wea.columns if c.startswith("timestamp")][0]
ti = pd.to_datetime(wea[tcol], errors="coerce")            # 若自带 -08:00 偏移，会直接变 tz-aware
ti = ti.dt.tz_localize(TZ) if ti.dt.tz is None else ti.dt.tz_convert(TZ)
wea = wea.set_index(ti).sort_index()
for c in ["ghi","dni","dhi","temp_air","wind_speed"]:
    if c not in wea.columns: raise ValueError(f"缺少列 {c}")
    wea[c] = to_float_series(wea[c])
if "albedo" not in wea.columns:
    wea["albedo"] = 0.2

wea_1m = wea[["ghi","dni","dhi","temp_air","wind_speed","albedo"]].resample("1min").interpolate("time")
wea_1m[["ghi","dni","dhi"]] = wea_1m[["ghi","dni","dhi"]].clip(lower=0)
wea_1m.to_csv(OUTDIR/"weather_1min_clean.csv", date_format="%Y-%m-%dT%H:%M:%S%z")


print("✅ 阶段A完成：已保存到 outputs/clean/ 下：pv_1min_clean.*、weather_1min_clean.*")
