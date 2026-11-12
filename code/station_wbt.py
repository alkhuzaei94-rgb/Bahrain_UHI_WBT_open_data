"""Station analysis: daily wet-bulb temperature (WBT) at Bahrain International Airport (WMO 41150), 2015-2024, from
daily mean air temperature and relative humidity; Table 3, the Figure 5b counts, trends and the comparison with NOAA
GSOD.

The daily records come from the Bahrain Meteorological Directorate and are not redistributed here; they are
available from the Directorate on request (monthly averages: Bahrain Open Data Portal, data.gov.bh). Give the two
month-by-day sheets as received. The days listed in data/station_days_replaced_noaa_gsod.csv are replaced with NOAA
GSOD values, which the script downloads from NOAA and checks against the list.
Run:    python code/station_wbt.py --temperature MeanDailyTemperature.xlsx --humidity MeanDailyRelativeHumidity.xlsx
Writes: results/station_summer_climate.csv (Table 3), wbt_days_above_thresholds.csv (Figure 5b), station_summary.csv
"""
import argparse, re, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
GSOD_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/41150099999.csv"
BLOCKS = (("2016-01-01", "2016-03-31"), ("2022-03-01", "2022-08-31"))   # replaced temperature / humidity blocks

ap = argparse.ArgumentParser()
ap.add_argument("--temperature", required=True, help="Directorate sheet of daily mean air temperature (°C)")
ap.add_argument("--humidity", required=True, help="Directorate sheet of daily mean relative humidity (%%)")
ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "bahrain_uhi_gsod"), help="NOAA GSOD download folder")
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
RES.mkdir(exist_ok=True)


def read_month_by_day(path):
    """Directorate sheet: one row per month (a year-month cell), one column per day 1-31. Returns date, value."""
    raw = pd.read_excel(path, header=None)
    tok = lambda v: str(int(v)) if isinstance(v, (int, float)) and not pd.isna(v) and float(v).is_integer() else str(v).strip()
    days = {str(i) for i in range(1, 32)}
    score = lambda i: sum(tok(c) in days for c in raw.iloc[i]) + 3 * any("year" in str(c).lower() for c in raw.iloc[i])
    hr = max(range(min(len(raw), 60)), key=score)
    df = pd.read_excel(path, header=hr)
    df.columns = [re.sub(r"^0?([1-9]|[12]\d|3[01])$", lambda m: str(int(m.group(1))), str(c).strip()) for c in df.columns]
    ym = next((c for c in df.columns if "year" in c.lower() or "month" in c.lower()), df.columns[0])
    dcols = [c for c in df.columns if c.isdigit() and 1 <= int(c) <= 31]
    long = df[[ym] + dcols].melt(id_vars=[ym], var_name="day", value_name="value")
    long["value"] = pd.to_numeric(long.value, errors="coerce")
    when = pd.to_datetime(long[ym] if np.issubdtype(long[ym].dtype, np.datetime64) else long[ym].astype(str), errors="coerce")
    long = long[when.notna() & long.value.notna()]                     # month rows with a value only
    when = when[long.index]
    date = pd.to_datetime(pd.DataFrame({"year": when.dt.year, "month": when.dt.month, "day": long.day.astype(int)}),
                          errors="coerce")                             # 30 February and the like become NaT
    return pd.Series(long.value.values, index=date.values).dropna().loc[lambda s: s.index.notna()].sort_index()


def stull(t, rh):
    """Stull (2011) wet-bulb temperature, Eq. (4) of the paper (T in °C, RH in %, valid -20 to 50 °C and 5 to 99 %)."""
    t, rh = np.clip(np.asarray(t, float), -20, 50), np.clip(np.asarray(rh, float), 5, 99)
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def gsod(year):
    f = Path(args.cache) / f"{year}.csv"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(GSOD_URL.format(year=year), timeout=120)
        r.raise_for_status()
        f.write_bytes(r.content)
    return pd.read_csv(f)


es = lambda t: 6.112 * np.exp(17.67 * t / (t + 243.5))              # Bolton (1980) saturation vapour pressure, hPa
g = pd.concat([gsod(y) for y in range(2015, 2025)])
g["date"] = pd.to_datetime(g.DATE)
g = g.set_index("date")
f2c = lambda f: (f - 32) * 5 / 9
both_ok = (g.TEMP < 999) & (g.DEWP < 999)
g_t = f2c(g.TEMP[both_ok])
g_rh = 100 * es(f2c(g.DEWP[both_ok])) / es(g_t)                     # daily RH from daily mean T and dew point

# ---------------------------------------------------------------- daily series with the listed NOAA days
t_dir, rh_dir = read_month_by_day(args.temperature), read_month_by_day(args.humidity)
rep = pd.read_csv(DATA / "station_days_replaced_noaa_gsod.csv", parse_dates=["date"])
noaa_now = np.where(rep.variable == "t_air_c", g_t.reindex(rep.date).values, g_rh.reindex(rep.date).values)
assert np.allclose(np.round(noaa_now, 3), rep.noaa_gsod_value, atol=1e-3), "NOAA values differ from the list"
t, rh = t_dir.copy(), rh_dir.copy()
for var, s in (("t_air_c", t), ("rh_pct", rh)):
    r = rep[rep.variable == var].set_index("date").noaa_gsod_value
    r = r[r.index.isin(s.index)]
    s.loc[r.index] = r.values
d = pd.DataFrame({"t_air_c": t, "rh_pct": rh}).dropna()
d["wbt_c"] = stull(d.t_air_c, d.rh_pct)
d["year"], d["month"] = d.index.year, d.index.month
a = d[d.year.between(2016, 2024)]
j = a[a.month.isin([6, 7, 8])]

# ---------------------------------------------------------------- Table 3 and Figure 5b
t3 = j.groupby("year").agg(t_air_mean_c=("t_air_c", "mean"), rh_mean_pct=("rh_pct", "mean"), wbt_mean_c=("wbt_c", "mean"),
                           wbt_peak_c=("wbt_c", "max"), days_wbt_above_30=("wbt_c", lambda v: int((v > 30).sum())))
t3.round(4).reset_index().to_csv(RES / "station_summer_climate.csv", index=False)
thr = pd.DataFrame([{"wbt_threshold_c": th, "days_above_2016_2024": int((a.wbt_c > th).sum()),
                     "days_per_year": (a.wbt_c > th).sum() / 9} for th in range(26, 33)])
thr.round(2).to_csv(RES / "wbt_days_above_thresholds.csv", index=False)

# ---------------------------------------------------------------- values quoted in the text
s = {"days_in_series_2015_2024": len(d), "days_replaced_temperature": int((rep.variable == "t_air_c").sum()),
     "days_replaced_humidity": int((rep.variable == "rh_pct").sum()),
     "summer_t_air_mean_c": j.t_air_c.mean(), "summer_rh_mean_pct": j.rh_pct.mean(), "summer_wbt_mean_c": j.wbt_c.mean(),
     "summer_rh_p05_pct": j.rh_pct.quantile(0.05), "summer_rh_p95_pct": j.rh_pct.quantile(0.95)}
for m, name in ((7, "july"), (8, "august")):
    mm = a[a.month == m]
    s[f"{name}_t_air_mean_c"], s[f"{name}_rh_mean_pct"], s[f"{name}_wbt_mean_c"] = mm.t_air_c.mean(), mm.rh_pct.mean(), mm.wbt_c.mean()
s["wbt_highest_daily_c"] = a.wbt_c.max()
s["wbt_highest_daily_date"] = a.wbt_c.idxmax().strftime("%Y-%m-%d")
s["days_wbt_above_30_in_june_august"] = int((j.wbt_c > 30).sum())
x = t3.index.values.astype(float)
for col, name in (("t_air_mean_c", "summer_t_air"), ("wbt_mean_c", "summer_wbt")):
    s[f"{name}_sen_per_decade"] = stats.theilslopes(t3[col].values, x)[0] * 10
    s[f"{name}_mk_p"] = stats.kendalltau(x, t3[col].values)[1]          # exact for n = 9 without ties
s["r_summer_wbt_vs_rh"] = t3.wbt_mean_c.corr(t3.rh_mean_pct)
s["r_summer_wbt_vs_t_air"] = t3.wbt_mean_c.corr(t3.t_air_mean_c)
# NOAA GSOD climate facts, 2016-2024
gg = g[(g.index.year >= 2016) & (g.index.year <= 2024)]
tmax = f2c(gg.MAX.where(gg.MAX < 999))
tmean = f2c(gg.TEMP.where(gg.TEMP < 999))
s["gsod_summer_daily_max_mean_c"] = tmax[tmax.index.month.isin([6, 7, 8])].mean()
s["gsod_daily_max_highest_c"] = tmax.max()
s["gsod_daily_max_highest_date"] = tmax.idxmax().strftime("%Y-%m-%d")
djf = tmean[tmean.index.month.isin([12, 1, 2])]
s["gsod_winter_daily_mean_p05_c"], s["gsod_winter_daily_mean_p95_c"] = djf.quantile(0.05), djf.quantile(0.95)
# agreement of the Directorate records with GSOD outside the two replaced blocks
cmp = pd.DataFrame({"t_dir": t_dir, "rh_dir": rh_dir}).join(pd.DataFrame({"t_g": g_t, "rh_g": g_rh}), how="inner").dropna()
cmp = cmp[(cmp.index >= "2015-01-01") & (cmp.index <= "2024-12-31")]
for lo, hi in BLOCKS:
    cmp = cmp[~((cmp.index >= lo) & (cmp.index <= hi))]
s["gsod_comparison_days"] = len(cmp)
s["gsod_t_mean_difference_c"] = (cmp.t_dir - cmp.t_g).mean()
s["gsod_t_abs_difference_p95_c"] = (cmp.t_dir - cmp.t_g).abs().quantile(0.95)
s["gsod_rh_mean_difference_pct"] = (cmp.rh_dir - cmp.rh_g).mean()
fmt = lambda v: v if isinstance(v, str) else (str(int(v)) if float(v).is_integer() else f"{float(v):.6g}")
pd.DataFrame({"quantity": list(s), "value": [fmt(v) for v in s.values()]}).to_csv(RES / "station_summary.csv", index=False)
print(t3.round(1).to_string())
print(thr.round(1).to_string(index=False))
for k, v in s.items():
    print(f"{k:<36} {fmt(v)}")
