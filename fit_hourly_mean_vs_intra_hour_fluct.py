# B_analyze_from_clean.py
# 功能：仅从阶段A保存的文件读取，完成 Perez 转置 → 温度修正 → 晴空归一化 → SWT 波动 → 制图
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import pvlib, pywt
from pvlib import location, irradiance, atmosphere, temperature

# ==== 基本参数 ====
TZ = "America/Los_Angeles"
LAT, LON, ALT = 37.42, -122.17, 30
TILT_DEG, AZM_DEG, ALBEDO_DEFAULT = 22.5, 195.0, 0.2
P_RATED_KW = 30.1
GAMMA_PDC = -0.0047                      # 25°C 参考的 PVWatts 温度系数（1/°C）
SAPM_A, SAPM_B, SAPM_dT = -3.47, -0.0594, 3.0
WAVELET, SWT_LEVELS = "db4", (3, 4)      # 8–32 min（D3+D4）
EPS = 1e-12
AUTO_FLIP_IF_NEGATIVE_CORR = True

INP = Path("outputs/clean"); OUT = Path("outputs"); OUT.mkdir(parents=True, exist_ok=True)

# 读回 CSV → 先统一解析成 tz-aware（UTC），再转本地时区（解决 DST 混合偏移）
def read_with_dtindex(path_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(path_csv, index_col=0)
    # 若保存时用 "%Y-%m-%dT%H:%M:%S%z"，这里用 utc=True 解析，再 tz_convert 到目标时区
    idx = pd.to_datetime(df.index, format="%Y-%m-%dT%H:%M:%S%z", utc=True, errors="coerce")
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError(f"{path_csv} 索引无法解析为 DatetimeIndex")
    df.index = idx.tz_convert(TZ)  # tz-aware 的安全转换
    return df

pv = read_with_dtindex(INP/"pv_1min_clean.csv")          # 列：P_kW
wea = read_with_dtindex(INP/"weather_1min_clean.csv")    # 列：ghi/dni/dhi/temp_air/wind_speed/albedo（若缺则稍后补默认）

# 对齐时段
idx = pv.index.intersection(wea.index)
pv = pv.reindex(idx); wea = wea.reindex(idx)
if "albedo" not in wea.columns:
    wea["albedo"] = ALBEDO_DEFAULT

# 太阳位置与晴空
site = location.Location(LAT, LON, tz=TZ, altitude=ALT)
sol  = site.get_solarposition(idx)
cs   = site.get_clearsky(idx, model="ineichen")

# Perez 转置到组件面（实际 & 晴空）— 必须提供 dni_extra，且 perez 需要相对气质量
def perez_poa(ghi, dni, dhi, solpos, albedo):
    dni_extra = irradiance.get_extra_radiation(ghi.index)                 # 外层直射 W/m^2
    airmass   = atmosphere.get_relative_airmass(solpos["apparent_zenith"])
    return irradiance.get_total_irradiance(
        surface_tilt=TILT_DEG, surface_azimuth=AZM_DEG,
        solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"],
        dni=dni, ghi=ghi, dhi=dhi,
        dni_extra=dni_extra, airmass=airmass,
        albedo=albedo, model="perez"
    )["poa_global"]

poa_act = perez_poa(wea["ghi"], wea["dni"], wea["dhi"], sol, wea["albedo"])
poa_cs  = perez_poa(cs["ghi"],   cs["dni"],   cs["dhi"],   sol, wea["albedo"])

# SAPM 电池温度 → 25°C 温度修正
Tcell  = temperature.sapm_cell(poa_act, wea["temp_air"], wea["wind_speed"],
                               a=SAPM_A, b=SAPM_B, deltaT=SAPM_dT)
P25_kW = pv["P_KW"] / (1.0 + GAMMA_PDC*(Tcell - 25.0))

# 常见口径：若日间与 GHI 负相关，自动翻转符号
r = P25_kW[wea["ghi"]>0].corr(wea["ghi"][wea["ghi"]>0])
if AUTO_FLIP_IF_NEGATIVE_CORR and pd.notna(r) and r < -0.2:
    P25_kW = -P25_kW

# 晴空归一化（逐时均值）：x = mean( P25 / P_cs )
P_cs_kW = P_RATED_KW * (poa_cs/1000.0)
x_inst  = (P25_kW.clip(lower=0) / (P_cs_kW + EPS)).clip(lower=0)
x_1h    = x_inst.resample("1h", label="right", closed="right").mean()

# SWT（Stationary Wavelet Transform，不下采样）提取 8–32min 波动 → 小时 σ
def swt_fluct(series_1m, levels=(3,4), wave="db4"):
    x = series_1m.values.astype(float); n=len(x); Lwant=max(levels); m=2**Lwant; r=n%m
    if r: L=(m-r)//2; R=(m-r)-L; xpad=np.pad(x,(L,R),mode="symmetric")
    else: L=R=0; xpad=x
    coeffs = pywt.swt(xpad, wavelet=wave, level=Lwant, norm=True)
    sel=[]
    for j,(cA,cD) in enumerate(coeffs, start=1):
        sel.append((np.zeros_like(cA), cD if j in set(levels) else np.zeros_like(cD)))
    y = pywt.iswt(sel, wavelet=wave)
    return pd.Series(y[L:L+n], index=series_1m.index)

P25_pu  = (P25_kW / (P_RATED_KW + EPS))
fluc_1m = swt_fluct(P25_pu, levels=SWT_LEVELS, wave=WAVELET)
sigma_1m = fluc_1m.resample("1h", label="right", closed="right").std(ddof=1)
sigma_5m = fluc_1m.resample("5min").mean().resample("1h", label="right", closed="right").std(ddof=1)

# 导出 + 作图（双对数，带分箱中位线）
out = pd.DataFrame({"x_cs": x_1h, "sigma_1m": sigma_1m, "sigma_5m": sigma_5m})
out.to_csv(OUT/"hourly_features_from_clean.csv", date_format="%Y-%m-%dT%H:%M:%S%z")

def scatter_loglog(x, y, title, png):
    df = pd.DataFrame({"x":x, "y":y}).dropna()
    if df.empty:
        print(f"[WARN] 无有效样本：{title}"); return
    x1,x2 = float(df["x"].quantile(0.01)), float(df["x"].quantile(0.99))
    y1,y2 = float(df["y"].quantile(0.01)), float(df["y"].quantile(0.99))
    x1=max(x1,1e-6); y1=max(y1,1e-8)
    edges = np.logspace(np.log10(x1), np.log10(max(x2, x1*10)), 16)
    df["bin"] = pd.cut(df["x"], bins=edges, include_lowest=True)
    med = df.groupby("bin").agg(x_mid=("x","median"), y_med=("y","median")).dropna()

    plt.figure(figsize=(6.8,4.8))
    plt.scatter(df["x"], df["y"], s=10, alpha=.28, label="samples")
    if not med.empty:
        plt.plot(med["x_mid"], med["y_med"], "-o", ms=4, lw=2, label="binned median")
    ax=plt.gca(); ax.set_xscale("log"); ax.set_yscale("log")
    plt.xlabel("x = 平均(P25 / P_cs)（log）"); plt.ylabel("σ (p.u., log)")
    plt.title(title); plt.grid(True, which="both", ls="--", alpha=.35); plt.legend()
    plt.tight_layout(); plt.savefig(OUT/png, dpi=180); plt.show()

scatter_loglog(out["x_cs"], out["sigma_1m"], "σ₁ₘ vs x（log–log, 8–32min）", "scatter_sigma1m_vs_xcs_loglog.png")
scatter_loglog(out["x_cs"], out["sigma_5m"], "σ₅ₘ vs x（log–log, 8–32min）", "scatter_sigma5m_vs_xcs_loglog.png")
print("✅ 阶段B完成：hourly_features_from_clean.csv 与两张散点图已输出到 outputs/")
