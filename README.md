# Nighttime urban heat and humid-heat stress in Bahrain: data and code

Data and code for the article "Multi-Scale Investigation of Nighttime Urban Heat and Humid-heat Stress in a Hot Desert
Climate" (Sustainable Cities and Society). Temperatures are in °C, areas in km².

| Path | Content |
|---|---|
| `code/satellite_lst_suhi.py` | Urban masks, land surface temperature, surface urban heat island, governorates, nighttime products, population against nighttime LST (Google Earth Engine) |
| `code/station_wbt.py` | Daily wet-bulb temperature at Bahrain International Airport, trends, comparison with NOAA GSOD |
| `code/field_sites.py` | Field measurements at the four neighborhoods |
| `data/study_area/` | Study-area polygon |
| `data/governorates/` | The four governorates |
| `data/station_days_replaced_noaa_gsod.csv` | The station days replaced with NOAA GSOD values |
| `data/field_measurements.xlsx` | Field measurements, 5-9 September, 18:00-22:00 |
| `results/` | The tables the scripts write (below) |

| Result | Article |
|---|---|
| `urban_extent.csv`, `threshold_sweep.csv` | Urban masks |
| `lst_suhi_monthly.csv`, `suhi_summer.csv` | Figure 3, SUHI |
| `night_products_summer.csv` | Nighttime products |
| `lst_by_governorate.csv` | Figure 4 |
| `station_summer_climate.csv` | Table 3 |
| `wbt_days_above_thresholds.csv` | Figure 5b |
| `field_site_means.csv`, `field_wbt_hourly.csv` | Table 4, Figure 6 |
| `satellite_summary.csv`, `station_summary.csv`, `field_summary.csv` | Other values quoted in the text |

## Data not included

The daily air temperature and relative humidity of Bahrain International Airport are records of the Bahrain
Meteorological Directorate, available from the Directorate on request; monthly averages are published on the Bahrain
Open Data Portal (data.gov.bh). The satellite data are open in Google Earth Engine; NOAA GSOD is downloaded by
`station_wbt.py`.

## Run

```
pip install -r requirements.txt
python code/satellite_lst_suhi.py --project YOUR_EARTH_ENGINE_PROJECT
python code/station_wbt.py --temperature MeanDailyTemperature.xlsx --humidity MeanDailyRelativeHumidity.xlsx
python code/field_sites.py
```

Tested with Python 3.13.

## Citation

Please cite the article above and this repository. Citation metadata is in `CITATION.cff`
(GitHub's "Cite this repository" button).

## Licence

[CC BY 4.0](LICENSE)
