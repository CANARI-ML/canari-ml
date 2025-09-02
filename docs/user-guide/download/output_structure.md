# Output Directory Structure

When you run the `canari_ml download` command, several directories and files are created in the `data/` folder, which is the root of all download outputs. This section explains the purpose of each of these folders and how the downloaded and processed data is organised.

## Overview

The downloader performs the following key steps:

- Downloads raw hourly ERA5 data from the [NSF NCAR AWS S3 mirror](https://registry.opendata.aws/nsf-ncar-era5/).
- Caches the raw files from the mirror under `data/aws/cache/`.
- Extracts the requested variables and pressure levels.
- Standardises variable names to match internal [download-toolbox](https://github.com/environmental-forecasting/download-toolbox) conventions.
- Saves daily-aggregated data to: `data/aws/day/north/{variable_name}_{pressure_level}/`

__

## Simplified Output Tree

``` console
data/
├── aws/
│   ├── cache/                # Raw hourly netCDF downloads from AWS S3 bucket
│   ├── day/
│   │   └── north/
│   │       ├── sic/          # Daily surface-level variable (e.g. sea ice concentration)
│   │       ├── tas/          # Daily surface-level variable (e.g. air temperature at 2m)
│   │       ├── tos/          # Daily sea surface temperature
│   │       ├── ua500/        # Daily upper-air wind at 500 hPa
│   │       ├── ua700/        # Daily upper-air wind at 700 hPa
│   │       ├── zg500/        # Daily geopotential height at 500 hPa
│   │       ├── zg700/        # Daily geopotential height at 700 hPa
│   │       └── <more_variables>/
│   ├── sic/                  # Full-resolution hourly file (netCDF) for surface variables
│   ├── tas/
│   ├── tos/
│   ├── ua500/
│   ├── ua700/
│   ├── zg500/
│   ├── zg700/
│   └── <more_variables>/
├── logs/
│   └── download_<timestamp>.log  # Log file for download operation
└── data.aws.DAY.north.json       # Metadata config file of downloads
```

???+ warning
    The ERA5 files from AWS contain all 37 pressure levels in a single file. Even if you only need one level (e.g. 500 hPa), the downloader retrieves the entire file once, extracts what you need, and reuses the file to avoid future re-downloads. You can set this cache to delete by using the `--delete-cache`.

### Output Breakdown

Temporary files: `data/aws/cache/`

- This folder contains the raw hourly ERA5 netCDF files as downloaded from the AWS mirror. These files are large (often ~1–1.5GB each) and are named by variable code and date range.
- Example: `e5.oper.an.pl.128_131_u.ll025uv.1979010100_1979010123.nc`

Temporary files: `data/aws/{variable_name}{pressure_level}/`

- These folders contain processed netCDFs for each variable and pressure level. Each one includes a single .nc file (e.g. 1979.nc) representing the full time period, typically at hourly resolution.
- Example: `data/aws/ua700/north/1979.nc`

**Persistent files**: `data/aws/day/north/{variable_name}_{pressure_level}/`

- This contains daily-aggregated versions of the same data for easier analysis.
- These are the final outputs that are used in this project, and **must not be deleted**.
- Example: `data/aws/day/north/ua700/1979.nc`

**Persistent files**:`data/aws/{surface_variable}/`

- Surface-level variables (e.g. sic, tas, tos) that do not require pressure level information are also saved here in both full-resolution and daily formats.

**Log files**: `logs/`

- Contains log files from each run of the downloader. Useful for debugging or confirming what was downloaded and processed.

**Config file**: `data.aws.DAY.north.json`

- An internal metadata file indexing all the daily-aggregated files, used by downstream components to track what’s available.

## Summary

After running the `canari_ml download` command with your specific overrides, by default, you will find:

- Raw ERA5 downloaded files → data/aws/cache/
- Hourly extracted files → `data/aws/{variable_name}{pressure_level}/`
- Daily aggregated files → `data/aws/day/north/{variable_name}_{pressure_level}/`
- Log files and metadata → `data/logs/`
- Metadata config file → `data.aws.DAY.north.json`

This structured layout helps separate raw, processed, and aggregated files — making it easy to plug into the rest of your ML or analysis pipelines.
