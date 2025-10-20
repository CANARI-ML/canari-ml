# Preprocess Output Structure

When you run the `canari_ml preprocess` command, all outputs are created under the root `preprocessed_data/` directory.
This section explains the purpose of each of these folders and how the downloaded and processed data is organised for training/prediction.

## Overview

The preprocessing pipeline performs the following key steps:

- Reprojects and masks the original ERA5 data (Northern Hemisphere only).
- Applies spatial masking for hemisphere and region-based weighting.
- Normalises the data, and converts variables such as Geopotential (`z`) to Geopotential Height (`zg`).
- Generates train/val/test splits.
- Caches the final datasets into Zarr format ready for training.
- Symlinks all the above stages into a single location for convenience.

__

## Simplified Output Tree

```bash
preprocessed_data/
├── cache/
│   └── 03_cache_{name}/       # Step 3: Final cached Zarr datasets (train/val/test)
│       ├── train_{name}/
│       │   ├── train/               # Training Zarr dataset
│       │   ├── val/                 # Validation Zarr dataset
│       │   └── test/                # Test Zarr dataset
│       └── cached.DAY.north.json    # Metadata config for cached data
│
├── preprocessed/
│   ├── 01_reproject_{name}/   # Step 1: Reproject raw ERA5 data to common EPSG:6931
│   │   ├── era5/                    # Reprojected files
│   │   │   └── day/north/
│   │   │       ├── sic/             # Sea ice concentration
│   │   │       ├── tas/             # 2m air temperature
│   │   │       ├── tos/             # Sea surface temperature
│   │   │       ├── ua500/           # Upper-air wind at 500 hPa
│   │   │       ├── ua700/           # Upper-air wind at 700 hPa
│   │   │       ├── zg500/           # Geopotential height at 500 hPa
│   │   │       └── zg700/           # Geopotential height at 700 hPa
│   │   ├── masks/
│   │   │   ├── hemisphere/          # Numpy array mask of Northern Hemisphere
│   │   │   └── weighted_regions/    # Weighted regional masks
│   │   └── reproject.DAY.north.json # Config file for this reprojection step
│   │
│   ├── 02_normalised_{name}/  # Step 2: Normalised data and masking
│   │   ├── era5/
│   │   │   ├── normalisation.scale/ # Per-variable normalisation parameters
│   │   │   ├── params/              # Climatology files (netCDF) used for anomaly calculation
│   │   │   ├── ua500_abs.nc
│   │   │   ├── ua700_abs.nc
│   │   │   ├── zg500_anom.nc
│   │   │   └── zg700_anom.nc
│   │   ├── mask_{name}/
│   │   │   ├── masks.north/         # Masks in netCDF format
│   │   │   ├── dataset_config.masks.DAY.north.json
│   │   │   └── processed_era5.masks.DAY.north.json
│   │   └── processed_era5.DAY.north.json # Config file for all normalising and mask implementation
│   │
│   └── loader.train_{name}.json  # Config file defining all preprocessed files and commands run
│
└── train_{name}/ # Main directory which symlinks to all the steps, and includes log files
    ├── 01_reproject_{name} -> ../preprocessed/01_reproject_{name}/
    ├── 02_normalised_{name} -> ../preprocessed/02_normalised_{name}/
    ├── 03_cache_{name} -> ../cache/03_cache_{name}/
    └── train_{name}_<timestamp>.log
```

Where `{name}` is the dataset name specified by the user, e.g. `input.name=train_demo_dataset`.
If `input.name` is not specified, it is set to label it based on a hash generated from the
dictionary items in `input`. This means that each time you run it, it would skip over
any steps that have already been completed.

### Output Breakdown

`preprocessed/01_reproject_{name}/`

- Reprojects ERA5 files to EPSG:6931 (Lambert Azimuthal Equal-Area for northern hemisphere).
- Generates hemisphere masking (to weight against training on Southern hemisphere).
    - Also includes `.npy` masks under `masks/` for hemisphere and region-based masking.
- Splits data into training, validation, and test sets.
- Stores all reprojected files under `era5/day/north/{variable_name}/`.

`preprocessed/02_normalised_{name}/`

- Normalises the data for training (or, if generating prediction dataset, using the trained dataset).
- Converts Geopotential (`z`) to Geopotential Height (`zg`).
- Stores absolute/anomaly of each variable.
- Rewrites masks to netCDF format.

`cache/03_cache_{name}/`

- Generates cached datasets in Zarr format.
- Split into `train/`, `val/`, and `test/` datasets.
- Each contains:
    - `x/`: Input variables
    - `y/`: Target variables
    - `time/`: Time variable
    - `sample_weights/`: Spatial weighting for loss functions to weight only Northern hemisphere (from given mask)
    - These are the datasets loaded during training.

`loader.{name}.json`

- JSON config file that defines paths to all prior steps: reprojected, normalised, masked, cached datasets.
- Required by the training code to know how to construct the dataset pipeline.

`train_{name}/ (symlinked convenience folder)`

- Convenience directory that symlinks all outputs into one place.
- Includes a log file from the full canari_ml preprocess run.
- Can be directly passed into training scripts (e.g., --data-dir train_{name}/).

## Summary

After running the `canari_ml preprocess` command:

- All outputs from this command are placed in the `preprocessed_data/` directory.
- The preprocessing includes reprojecting, normalising, masking, and caching (caching only if generating training dataset).
- Final training-ready data is accessible in Zarr format under `cache/03_cache_{name}/` along with the corresponding config file in the same directory.
    - If prediction-ready data is required, it is accessible by directly referencing the JSON config file within the same directory as above.
        - In this case, the config file is created, but corresponding Zarr is not.

This approach ensures complete reproducibility of how each dataset was built, and keeps intermediate and final files clearly separated for as per the Unix philosophy.
