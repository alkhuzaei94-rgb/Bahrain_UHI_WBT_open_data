"""Satellite analysis: annual urban masks, land surface temperature (LST), surface urban heat island (SUHI), governorate
statistics, the nighttime product cross-check, and population against nighttime LST. Bahrain, June-August 2016-2024.

Inputs: open data in Google Earth Engine (Landsat 8/9 Collection 2 Level-2, MODIS MYD11A1 and MYD21A1N, VIIRS
VNP21A1N, Dynamic World V1, ESA WorldCover v200, JRC GHSL P2023A) and the polygons in data/.
Run:    python code/satellite_lst_suhi.py --project YOUR_EARTH_ENGINE_PROJECT        (about 30 minutes)
Writes: results/urban_extent.csv, threshold_sweep.csv, lst_suhi_monthly.csv, suhi_summer.csv,
        night_products_summer.csv, lst_by_governorate.csv, satellite_summary.csv
"""
import argparse, calendar, sys, tempfile, time
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.warp import Resampling, reproject
from scipy import stats
from shapely.geometry import mapping
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
YEARS = list(range(2016, 2025))
WARMUP = 2015                       # seeds the two-detection rule
THRESHOLDS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)

ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True, help="Google Earth Engine project id")
ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "bahrain_uhi_rasters"),
                help="folder for the three exported rasters used by the population correlation")
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
ee.Initialize(project=args.project)
RES.mkdir(exist_ok=True)
t0 = time.time()
log = lambda msg: print(f"[{time.time() - t0:5.0f} s] {msg}", flush=True)

# ---------------------------------------------------------------- study area, land and reference classes
sa = gpd.read_file(DATA / "study_area" / "study_area.shp").to_crs(4326)
roi = ee.Geometry(mapping(unary_union(sa.geometry)))
wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").clip(roi)
LAND = wc.neq(80)
WC_BUILT = wc.eq(50).unmask(0).updateMask(LAND).rename("urban").toInt()
BARE = wc.eq(60).updateMask(LAND)
VEG = wc.eq(10).Or(wc.eq(20)).Or(wc.eq(30)).Or(wc.eq(40)).updateMask(LAND)
P10 = wc.projection()


def km2(mask):
    """Area of a mask: sum of the 10 m pixel areas."""
    a = (ee.Image.pixelArea().updateMask(mask.reproject(P10).selfMask())
         .reduceRegion(ee.Reducer.sum(), roi, 10, maxPixels=1e13, tileScale=4).get("area"))
    return float(ee.Number(a).getInfo()) / 1e6


def accuracy(mask):
    """Precision, recall and F1 of a mask against WorldCover built-up, pixel by pixel at 10 m."""
    cls = mask.multiply(2).add(WC_BUILT).rename("cls").reproject(P10)
    h = cls.reduceRegion(ee.Reducer.frequencyHistogram(), roi, 10, maxPixels=1e13, tileScale=4).get("cls").getInfo()
    h = {int(float(k)): v for k, v in h.items()}
    tp, fp, fn = h.get(3, 0), h.get(2, 0), h.get(1, 0)
    p, r = tp / (tp + fp), tp / (tp + fn)
    return p, r, 2 * p * r / (p + r)


def dw_year(y, th):
    """Pixels whose June-August median Dynamic World 'built' probability exceeds th."""
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(roi)
          .filterDate(f"{y}-06-01", f"{y}-08-31").select("built"))
    return dw.median().gt(th).unmask(0).updateMask(LAND).clip(roi).rename("urban").toInt()


summary = {}
summary["land_km2"] = km2(LAND)
summary["worldcover_built_2021_km2"] = km2(WC_BUILT)
ghs = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020").select("built_surface").clip(roi)
summary["ghsl_built_2020_km2"] = float(ee.Number(ee.Image.pixelArea().updateMask(ghs.gt(0)).reduceRegion(
    ee.Reducer.sum(), roi, 100, maxPixels=1e13).get("area")).getInfo()) / 1e6
log(f"land {summary['land_km2']:.2f} km2, WorldCover built-up {summary['worldcover_built_2021_km2']:.2f} km2")

# ---------------------------------------------------------------- threshold: 2021 extent closest to WorldCover
sweep = []
for th in THRESHOLDS:
    m = dw_year(2021, th)
    p, r, f1 = accuracy(m)
    sweep.append({"threshold": th, "dw_2021_km2": km2(m), "precision": p, "recall": r, "f1": f1})
    log(f"threshold {th:.2f}: {sweep[-1]['dw_2021_km2']:.1f} km2, F1 {f1:.3f}")
sweep = pd.DataFrame(sweep)
sweep["diff_to_worldcover_km2"] = (sweep.dw_2021_km2 - summary["worldcover_built_2021_km2"]).abs()
sweep.round(4).to_csv(RES / "threshold_sweep.csv", index=False)
best = sweep.loc[sweep.diff_to_worldcover_km2.idxmin()]
TH = float(best.threshold)
summary.update({"threshold": TH, "dw_2021_km2": best.dw_2021_km2, "precision_2021": best.precision,
                "recall_2021": best.recall, "f1_2021": best.f1})

# ---------------------------------------------------------------- annual urban masks (urban once detected twice)
count = ee.Image.constant(0).clip(roi).rename("count").add(dw_year(WARMUP, TH).rename("count"))
MASKS, ext = {}, []
for y in YEARS:
    count = count.add(dw_year(y, TH).rename("count"))
    urban = count.gte(2).unmask(0).updateMask(LAND).clip(roi).rename("urban").toInt()
    MASKS[y] = (urban, urban.Not().updateMask(LAND).clip(roi).rename("rural").toInt())
    ext.append({"year": y, "urban_km2": km2(urban)})
ext = pd.DataFrame(ext)
ext["non_urban_km2"] = summary["land_km2"] - ext.urban_km2
ext.round(3).to_csv(RES / "urban_extent.csv", index=False)
summary["urban_2016_km2"], summary["urban_2024_km2"] = ext.urban_km2.iloc[0], ext.urban_km2.iloc[-1]
summary["urban_growth_pct"] = 100 * (ext.urban_km2.iloc[-1] / ext.urban_km2.iloc[0] - 1)
log("urban masks done")


# ---------------------------------------------------------------- LST collections
def landsat_lst(img):
    """Landsat C2 L2 ST_B10 to °C; QA_PIXEL bits 0-5 (fill, cloud, shadow, snow) and ST_QA > 3 K removed."""
    ok = img.select("QA_PIXEL").bitwiseAnd(0b111111).eq(0)
    lst = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("LST")
    good = img.select("ST_QA").multiply(0.01).lte(3.0)
    return lst.updateMask(ok).updateMask(good).copyProperties(img, ["system:time_start"])


def myd11(img):
    """MYD11A1 nighttime LST to °C; QC_Night 'good' or 'other quality'."""
    keep = img.select("QC_Night").bitwiseAnd(3).lte(1)
    return (img.select("LST_Night_1km").multiply(0.02).subtract(273.15).rename("LST").updateMask(keep)
            .copyProperties(img, ["system:time_start"]))


def tes(img):
    """MYD21A1N and VNP21A1N: LST_1KM is already in kelvin."""
    keep = img.select("QC").bitwiseAnd(3).lte(1)
    return img.select("LST_1KM").subtract(273.15).rename("LST").updateMask(keep).copyProperties(img, ["system:time_start"])


def coll(cid, fn):
    return ee.ImageCollection(cid).filterBounds(roi).filterDate("2016-01-01", "2024-12-31").map(fn)


LS = coll("LANDSAT/LC08/C02/T1_L2", landsat_lst).merge(coll("LANDSAT/LC09/C02/T1_L2", landsat_lst))
M11, M21, VN = coll("MODIS/061/MYD11A1", myd11), coll("MODIS/061/MYD21A1N", tes), coll("NASA/VIIRS/002/VNP21A1N", tes)


def zonal(img, mask, scale):
    return img.updateMask(mask).reduceRegion(ee.Reducer.mean(), roi, scale, maxPixels=1e13, bestEffort=True,
                                             tileScale=4).get("LST")


def fetch(feats):
    """All features of a year in one request; part by part if Earth Engine runs out of memory."""
    try:
        return [f["properties"] for f in ee.FeatureCollection(feats).getInfo()["features"]]
    except ee.EEException as e:
        if "memory" not in str(e).lower():
            raise
        log("memory limit reached: fetching the parts one by one")
        return [f.getInfo()["properties"] for f in feats]


# ---------------------------------------------------------------- monthly composites and SUHI; night products
rows, xc = [], []
for y in YEARS:
    urban, rural = MASKS[y]
    feats = []
    for m in (6, 7, 8):
        start, end = f"{y}-{m:02d}-01", f"{y}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"
        day = LS.filterDate(start, end).median().clip(roi).updateMask(LAND)
        night = M11.filterDate(start, end).median().clip(roi).updateMask(LAND)
        p = {"year": y, "month": m}
        for lab, msk in (("land", LAND), ("urban", urban), ("rural", rural), ("bare", BARE), ("veg", VEG)):
            p[f"day_{lab}"] = zonal(day, msk, 120)
        for lab, msk in (("urban", urban), ("rural", rural), ("bare", BARE), ("veg", VEG)):
            p[f"night_{lab}"] = zonal(night, msk, 1000)
        feats.append(ee.Feature(None, p))
    p = {"year": y, "month": 0}                              # one June-August composite per product
    for lab, ic in (("myd11a1", M11), ("myd21a1n", M21), ("vnp21a1n", VN)):
        img = ic.filterDate(f"{y}-06-01", f"{y}-08-31").median().clip(roi).updateMask(LAND)
        p[f"{lab}_urban"] = img.updateMask(urban).reduceRegion(ee.Reducer.mean(), roi, 1000, tileScale=4).get("LST")
        p[f"{lab}_rural"] = img.updateMask(rural).reduceRegion(ee.Reducer.mean(), roi, 1000, tileScale=4).get("LST")
    feats.append(ee.Feature(None, p))
    got = fetch(feats)
    rows += [g for g in got if g["month"] > 0]
    xc += [g for g in got if g["month"] == 0]
    log(f"year {y} done")

mo = pd.DataFrame(rows)
mo["day_rural"] = mo.day_rural.fillna(mo.day_land)          # no valid rural pixel: land-wide mean
mo["suhi_day"] = mo.day_urban - mo.day_rural
mo["suhi_night"] = mo.night_urban - mo.night_rural
cols = ["year", "month", "day_land", "day_urban", "day_rural", "day_bare", "day_veg",
        "night_urban", "night_rural", "night_bare", "night_veg", "suhi_day", "suhi_night"]
mo[cols].round(4).to_csv(RES / "lst_suhi_monthly.csv", index=False)

mo["day_urban_minus_bare"] = mo.day_urban - mo.day_bare
mo["day_urban_minus_veg"] = mo.day_urban - mo.day_veg
mo["night_urban_minus_bare"] = mo.night_urban - mo.night_bare
mo["night_urban_minus_veg"] = mo.night_urban - mo.night_veg
yr = mo.groupby("year")[["suhi_day", "suhi_night", "day_urban", "day_rural", "day_urban_minus_bare",
                         "day_urban_minus_veg", "night_urban_minus_bare", "night_urban_minus_veg"]].mean()
yr.round(4).reset_index().to_csv(RES / "suhi_summer.csv", index=False)

xc = pd.DataFrame(xc).set_index("year")
for lab in ("myd11a1", "myd21a1n", "vnp21a1n"):
    xc[f"{lab}_suhi"] = xc[f"{lab}_urban"] - xc[f"{lab}_rural"]
xc = xc[[f"{lab}_suhi" for lab in ("myd11a1", "myd21a1n", "vnp21a1n")]]
xc.round(4).reset_index().to_csv(RES / "night_products_summer.csv", index=False)

for c in yr.columns:
    summary[f"{c}_mean"] = yr[c].mean()
summary["suhi_day_min_year"], summary["suhi_day_max_year"] = yr.suhi_day.min(), yr.suhi_day.max()
for c in xc.columns:
    summary[f"{c}_mean"] = xc[c].mean()


def trend(series):
    """Sen's slope per decade and two-sided Mann-Kendall p (exact distribution of S, or the normal approximation
    with ties)."""
    x = series.index.values.astype(float)
    return stats.theilslopes(series.values, x)[0] * 10, stats.kendalltau(x, series.values)[1]


for name, s in (("suhi_day", yr.suhi_day), ("suhi_night", yr.suhi_night),
                ("myd11a1_suhi", xc.myd11a1_suhi), ("myd21a1n_suhi", xc.myd21a1n_suhi), ("vnp21a1n_suhi", xc.vnp21a1n_suhi)):
    summary[f"{name}_sen_per_decade"], summary[f"{name}_mk_p"] = trend(s)
summary["r_myd11a1_myd21a1n"] = xc.myd11a1_suhi.corr(xc.myd21a1n_suhi)
summary["r_myd11a1_vnp21a1n"] = xc.myd11a1_suhi.corr(xc.vnp21a1n_suhi)
summary["r_myd21a1n_vnp21a1n"] = xc.myd21a1n_suhi.corr(xc.vnp21a1n_suhi)

# ---------------------------------------------------------------- nine-year composite: island and governorates
comp = ee.ImageCollection([LS.filterDate(f"{y}-06-01", f"{y}-08-31").median().clip(roi).updateMask(LAND)
                           for y in YEARS]).mean()
summary["island_day_lst_composite"] = float(ee.Number(comp.reduceRegion(
    ee.Reducer.mean(), roi, 120, maxPixels=1e13, bestEffort=True).get("LST")).getInfo())
gov = gpd.read_file(DATA / "governorates" / "governorates.shp").to_crs(4326).dissolve(by="name")
u24, r24 = MASKS[2024]
grows = []
for name, row in gov.iterrows():
    g = row.geometry.simplify(0.0005, preserve_topology=True)
    polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
    eg = ee.Geometry.MultiPolygon([[[[float(c[0]), float(c[1])] for c in p.exterior.coords]] for p in polys],
                                  proj=None, geodesic=False)
    v = {lab: float(ee.Number(img.reduceRegion(ee.Reducer.mean(), eg, 120, maxPixels=1e13, bestEffort=True)
                              .get("LST")).getInfo())
         for lab, img in (("all_land", comp), ("urban", comp.updateMask(u24)), ("non_urban", comp.updateMask(r24)))}
    grows.append({"governorate": name, "lst_all_land": v["all_land"], "lst_urban": v["urban"],
                  "lst_non_urban": v["non_urban"], "urban_minus_non_urban": v["urban"] - v["non_urban"]})
pd.DataFrame(grows).round(4).to_csv(RES / "lst_by_governorate.csv", index=False)
log("composite and governorates done")

# ---------------------------------------------------------------- population against nighttime LST (1 km cells)
cache = Path(args.cache)
cache.mkdir(parents=True, exist_ok=True)


def download(img, name, scale):
    out = cache / f"{name}.tif"
    url = img.getDownloadURL({"region": roi, "scale": scale, "format": "GEO_TIFF", "crs": "EPSG:4326"})
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


night_ic = M11.filterDate(f"{YEARS[0]}-06-01", f"{YEARS[0]}-09-01")       # every summer night, 2016-2024
for y in YEARS[1:]:
    night_ic = night_ic.merge(M11.filterDate(f"{y}-06-01", f"{y}-09-01"))
f_night = download(night_ic.mean().rename("LST").clip(roi), "lst_night_summer_mean", 1000)
f_day = download(comp.rename("LST").updateMask(LAND).clip(roi), "lst_day_summer_mean", 100)
f_pop = download(ee.Image("JRC/GHSL/P2023A/GHS_POP/2020").select("population_count").updateMask(LAND).clip(roi),
                 "ghs_pop_2020", 100)
with rasterio.open(f_night) as s:
    night = s.read(1).astype(float)
    grid = dict(dst_transform=s.transform, dst_crs=s.crs)
night[night == 0] = np.nan
with rasterio.open(f_pop) as s:
    pop = s.read(1).astype(float)
    pop[~np.isfinite(pop) | (pop < 0)] = 0
    pop1k = np.zeros(night.shape)
    reproject(pop, pop1k, src_transform=s.transform, src_crs=s.crs, resampling=Resampling.sum, **grid)
with rasterio.open(f_day) as s:
    day = s.read(1).astype("float32")
    day[day == 0] = np.nan
    day1k = np.full(night.shape, np.nan, dtype="float32")
    reproject(day, day1k, src_transform=s.transform, src_crs=s.crs, resampling=Resampling.average,
              src_nodata=np.nan, dst_nodata=np.nan, **grid)
for lab, lst in (("night", night), ("day", day1k)):
    m = np.isfinite(night) & np.isfinite(lst)
    x, v = pop1k[m], lst[m]
    rho, p = stats.spearmanr(x, v)
    summary[f"pop_vs_{lab}_lst_cells"] = int(m.sum())
    summary[f"pop_vs_{lab}_lst_spearman"], summary[f"pop_vs_{lab}_lst_p"] = rho, p
    summary[f"{lab}_lst_cells_pop_ge_1000"] = v[x >= 1000].mean()
    summary[f"{lab}_lst_cells_pop_lt_100"] = v[x < 100].mean()
log("population correlation done")

pd.DataFrame({"quantity": list(summary), "value": [float(v) for v in summary.values()]}).to_csv(
    RES / "satellite_summary.csv", index=False, float_format="%.6g")
log("written: " + ", ".join(sorted(p.name for p in RES.glob("*.csv"))))
