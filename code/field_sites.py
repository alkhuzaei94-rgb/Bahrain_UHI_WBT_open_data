"""Field measurements at the four residential neighborhoods (5-9 September, 18:00-22:00): the on-site means of
Table 4 and the hourly wet-bulb temperature (WBT) of Figure 6, from data/field_measurements.xlsx.
In that file, surfaces are hourly readings by orientation; 'AT (Local)' and 'RH (Local)' are hourly means of the
10-minute logger readings, and 'WBGT (Local)' is the hourly mean of the 10-minute WBT from Eq. (4).
Run:    python code/field_sites.py
Writes: results/field_site_means.csv (Table 4), field_wbt_hourly.csv (Figure 6), field_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SITES = {"Inland HD": "Inland high density", "Inland LD": "Inland low density",
         "Coastal HD": "Coastal high density", "Coastal LD": "Coastal low density"}
COLS = {"roads_c": range(2, 6), "pavements_c": range(6, 10), "walls_c": range(10, 14),     # B:E, F:I, J:M
        "air_temperature_c": [18], "relative_humidity_pct": [19], "wbt_c": [20]}            # R, S, T
sys.stdout.reconfigure(encoding="utf-8")
RES.mkdir(exist_ok=True)

ws = load_workbook(ROOT / "data" / "field_measurements.xlsx", data_only=True)["AT_Classification per Location"]
num = lambda v: np.nan if v in (None, "") else float(v)
means, hourly = [], {}
for r in range(1, ws.max_row + 1):
    title = str(ws.cell(r, 1).value or "").strip()
    if title not in SITES:
        continue
    hdr = [ws.cell(r + 3, c).value for c in (2, 18, 19, 20)]
    assert hdr == ["North", "AT (Local)", "RH (Local)", "WBGT (Local)"], (title, hdr)
    rows = range(r + 4, r + 9)                                       # 18:00 to 22:00
    vals = {k: np.array([[num(ws.cell(rr, c).value) for c in cols] for rr in rows]) for k, cols in COLS.items()}
    means.append({"site": SITES[title], **{k: np.nanmean(v) for k, v in vals.items()}})
    hourly[SITES[title]] = vals["wbt_c"][:, 0]
    times = [ws.cell(rr, 1).value.strftime("%H:%M") for rr in rows]
order = ["Inland high density", "Inland low density", "Coastal high density", "Coastal low density"]
m = pd.DataFrame(means).set_index("site").loc[order]
m.round(3).reset_index().to_csv(RES / "field_site_means.csv", index=False)
h = pd.DataFrame({"local_time": times, **{s: hourly[s] for s in order}})
h.to_csv(RES / "field_wbt_hourly.csv", index=False, float_format="%.6f")

gap = h["Coastal high density"] - h["Coastal low density"]
summary = {"coastal_wbt_gap_mean_c": m.loc["Coastal high density", "wbt_c"] - m.loc["Coastal low density", "wbt_c"],
           "coastal_wbt_gap_hourly_max_c": gap.max(),
           "coastal_wbt_gap_hourly_max_hour": float(h.local_time[gap.idxmax()][:2]),
           "coastal_wbt_gap_2200_c": gap.iloc[-1],
           "inland_wbt_gap_mean_c": m.loc["Inland high density", "wbt_c"] - m.loc["Inland low density", "wbt_c"]}
pd.DataFrame({"quantity": list(summary), "value": list(summary.values())}).to_csv(
    RES / "field_summary.csv", index=False, float_format="%.6g")
print(m.round(2).to_string())
print(h.round(2).to_string(index=False))
for k, v in summary.items():
    print(f"{k:<34} {v:.3f}")
