# -*- coding: utf-8 -*-
"""
【混合版】
单点 GFS 0.25° → 仅取“前一日18Z起报 + 次日所需的3小时时效(015..039)”。
- wgrib2 -lon 提取除 TCDC 外的变量
- cfgrib 提取 TCDC 变量（因为它在 wgrib2 -lon 上会崩溃）
- 并行汇总到 1H 表；与历史功率合并。
依赖：wgrib2, pandas, numpy, tqdm, cfgrib, xarray
"""

import os, re, subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
from multiprocessing import cpu_count

# 【新增】为 TCDC 导入 cfgrib
try:
    import xarray as xr
    import cfgrib  # noqa: F401
except Exception as e:
    print(f"警告：无法导入 cfgrib/xarray ({e})。TCDC 变量将无法被提取。")
    xr = None
    cfgrib = None


# ------------------ 用户配置 ------------------
GRIB_DIR = r"G:\data\GFS\Stanford_2019"      # GRIB2 文件夹
OBS_CSV  = r"2019_pv_raw.csv"                # 历史功率文件（5/15min）
OBS_TIME_COL = "Timestemp"                  # 你的时间列名
OBS_VAL_COL  = "Power (kW)"                 # 你的功率列名
LOCAL_TZ = "America/Los_Angeles"
OUT_CSV  = r"outputs\pv_1h_obs_with_gfs18z.csv"

# 站点经纬度（注意：wgrib2 -lon 经度使用 0..360）
LAT = 37.4275
LON = -122.1697
LON360 = LON if LON >= 0 else LON + 360.0

# wgrib2 可执行文件路径
WGRIB2 = r".\wgrib2.exe"

# 仅保留 18Z 起报 & f015..f039（覆盖“次日”24h）
FILE_PAT = re.compile(
    r"^gfs\.0p25\.(\d{8})18\.f0(15|18|21|24|27|30|33|36|39)\.grib2$"
)

# 【修改】wgrib2 变量筛选 (TCDC 已移除)
MATCH_REGEX = (
    r":(DSWRF|DLWRF):surface:|"
    r":TMP:2 m above ground:|"
    r":RH:2 m above ground:|"
    r":UGRD:10 m above ground:|"
    r":VGRD:10 m above ground:"
)
VAR_MAP = {
    ("DSWRF","surface"): "dswrf",
    ("DLWRF","surface"): "dlwrf",
    ("TMP","2 m above ground"): "t2m",
    ("RH","2 m above ground"): "rh2m",
    ("UGRD","10 m above ground"): "u10",
    ("VGRD","10 m above ground"): "v10",
}
VAL_RE = re.compile(r"val=([\-0-9.eE]+)")

# ------------------ 工具函数 ------------------

def parse_name(fname: str):
    """从 gfs.0p25.YYYYMMDDHH.f###.grib2 提取起报与时效"""
    m = re.search(r"gfs\.\w+\.(\d{10})\.f(\d{3})\.grib2$", os.path.basename(fname))
    if not m:
        return None, None
    init_str, lead_str = m.groups()
    init_utc = pd.to_datetime(init_str, format="%Y%m%d%H", utc=True)
    return init_utc, int(lead_str)

# 【新增】一个专门用 cfgrib 提取 TCDC 的函数
def read_tcdc_with_cfgrib(path: str):
    """
    使用 cfgrib 专门打开 'entireAtmosphere' 组来提取 TCDC。
    根据我们的诊断，cfgrib 应该能成功打开这个组。
    """
    if cfgrib is None: # 检查 cfgrib 是否成功导入
        return np.nan
        
    try:
        bk = {"filter_by_keys": {"typeOfLevel": "entireAtmosphere"}}
        with xr.open_dataset(path, engine="cfgrib", 
                             backend_kwargs={**bk, "indexpath": ""}) as ds:
            
            for v_name in ds.data_vars:
                a = ds[v_name].attrs
                sn = str(a.get("GRIB_shortName","")).lower()
                if sn in {"tcdc", "tcc"}:
                    return float(np.asarray(ds[v_name].values).squeeze())
        return np.nan # 组已打开，但未找到 TCDC
    except Exception:
        return np.nan # 无法打开 'entireAtmosphere' 组

def run_wgrib2_point(path: str):
    """
    【修改】
    1. 用 wgrib2 提取除 TCDC 外的变量。
    2. 用 cfgrib 提取 TCDC 变量。
    """
    try:
        init_utc, lead_h = parse_name(path)
        if init_utc is None:
            return {"error": "Failed to parse name", "path": path}
        
        valid_utc = init_utc + pd.Timedelta(hours=lead_h)
        cmd = [
            WGRIB2, path,
            "-match", MATCH_REGEX,
            "-s",
            "-lon", f"{LON360}", f"{LAT}",
        ]
        
        cwd = os.path.dirname(os.path.abspath(WGRIB2))
        exe_name = os.path.basename(WGRIB2)
        cmd[0] = exe_name 
        
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, 
                             encoding='utf-8', errors='ignore',
                             cwd=cwd, shell=True) 

        if res.returncode != 0:
            if '3221225781' in str(res.returncode):
                 return {"error": f"DLL Not Found (code 3221225781). 确保所有 .dll 文件与 {WGRIB2} 在同一目录。", "stderr": res.stderr[:200], "path": path}
            return {"error": f"wgrib2 failed (code {res.returncode})", "stderr": res.stderr[:200], "path": path}

        out = res.stdout
        row = {"init_utc": init_utc, "valid_utc": valid_utc, "lead_hour": lead_h}
        if not out:
            return {"error": "wgrib2 ran but found no matching variables", "path": path}

        found_any = False
        for line in out.splitlines():
            parts = line.split(":")
            if len(parts) < 6: continue
            var = parts[3].strip()
            level = parts[4].strip()
            key = VAR_MAP.get((var, level))
            if not key: continue
            m = VAL_RE.search(line)
            if not m: continue
            try:
                row[key] = float(m.group(1))
                found_any = True
            except Exception:
                pass
        
        if not found_any:
            return {"error": "wgrib2 ran, output lines, but failed to parse value", "path": path}
        
        # 【新增】wgrib2 成功后，调用 cfgrib 提取 TCDC
        tcdc_val = read_tcdc_with_cfgrib(path)
        row["tcdc"] = tcdc_val # 添加 tcdc 值 (可能是 np.nan)
        
        return row

    except FileNotFoundError:
        return {"error": f"'{WGRIB2}' not found. 请检查 WGRIB2 变量的路径。", "path": path}
    except Exception as e:
        return {"error": f"Python error in worker: {str(e)}", "path": path}

# ------------------ 主执行逻辑 ------------------
def main():
    print(f"Finding files in: {GRIB_DIR}")
    print(f"Using pattern: {FILE_PAT.pattern}")
    
    all_files = [f for f in sorted(os.listdir(GRIB_DIR)) if FILE_PAT.match(f)]
    files = [os.path.join(GRIB_DIR, f) for f in all_files]
    if not files:
        raise SystemExit("未匹配到 18Z & f015..f039 的 GRIB2 文件，请检查目录与文件命名。")
    print(f"Found {len(files)} files to process.")
    
    wgrib2_full_path = os.path.abspath(WGRIB2)
    if not os.path.exists(wgrib2_full_path):
        print(f"!!! 致命错误: 找不到 wgrib2.exe")
        print(f"    检查路径: {wgrib2_full_path}")
        print(f"    (基于您的配置: WGRIB2 = r\"{WGRIB2}\")")
        raise SystemExit("wgrib2.exe 未找到。")
    print(f"Using wgrib2 at: {wgrib2_full_path}")
    if cfgrib is None:
        print("警告: cfgrib/xarray 未加载。TCDC 变量将全部为 NaN。")
        print("请运行: conda install -c conda-forge cfgrib python-eccodes xarray")


    rows = []
    failed_files = []
    max_workers = min(8, cpu_count() or 4)
    print(f"Starting ProcessPoolExecutor with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_wgrib2_point, p): p for p in files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="wgrib2/cfgrib hybrid"):
            r = fut.result()
            if isinstance(r, dict) and "error" in r:
                failed_files.append(r)
            elif r:
                rows.append(r)
                
    if failed_files:
        print(f"\n--- 警告：{len(failed_files)} / {len(files)} 个文件处理失败 ---")
        print(f"第一个错误示例: {failed_files[0]}")
        print("--------------------------------------\n")
    
    if not rows:
         raise SystemExit("所有文件都处理失败了，请检查上面的错误日志。")

    print(f"Successfully processed {len(rows)} files. Building DataFrame...")
    fc = (pd.DataFrame(rows)
          .drop_duplicates(subset=["init_utc","valid_utc"])
          .sort_values("valid_utc"))

    fc["valid_local"] = fc["valid_utc"].dt.tz_convert(LOCAL_TZ)
    fc["ts_hour_local"] = fc["valid_local"].dt.floor("1H")
    if {"u10","v10"}.issubset(fc.columns):
        fc["ws10"] = np.hypot(fc["u10"], fc["v10"])
    if "t2m" in fc.columns:
        fc["t2m_c"] = fc["t2m"] - 273.15

    print("Reading and resampling observations...")
    obs = pd.read_csv(OBS_CSV, parse_dates=[OBS_TIME_COL])
    if getattr(obs[OBS_TIME_COL].dt, "tz", None) is None:
        obs[OBS_TIME_COL] = obs[OBS_TIME_COL].dt.tz_localize(LOCAL_TZ)
    else:
        obs[OBS_TIME_COL] = obs[OBS_TIME_COL].dt.tz_convert(LOCAL_TZ)
    obs = obs.set_index(OBS_TIME_COL).sort_index()
    obs_h = obs[[OBS_VAL_COL]].resample("1H").mean().rename(columns={OBS_VAL_COL:"power_actual_1h"})

    print("Merging forecasts with observations...")
    idx = obs_h.index
    local_midnight = idx.tz_convert(LOCAL_TZ).normalize()
    target_init_utc = (local_midnight.tz_convert("UTC") - pd.Timedelta(hours=6))
    key_df = pd.DataFrame({"ts_hour_local": idx, "target_init_utc": target_init_utc.values})

    keep_cols = [c for c in fc.columns if c not in {"valid_utc","valid_local", "error", "path"}]
    fc_key = fc[keep_cols]
    merged = (key_df
              .merge(fc_key, left_on=["ts_hour_local","target_init_utc"],
                               right_on=["ts_hour_local","init_utc"],
                               how="left")
              .set_index("ts_hour_local"))
    final = obs_h.join(merged.drop(columns=["target_init_utc","init_utc"], errors='ignore'), how="left")

    print("Interpolating 3-hourly data to 1-hour...")
    num_cols = final.select_dtypes(include=[np.number]).columns
    final[num_cols] = final[num_cols].interpolate(method="time", limit_direction="both")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    final.to_csv(OUT_CSV, index_label="timestamp_local")
    print("\nOK →", OUT_CSV)

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()