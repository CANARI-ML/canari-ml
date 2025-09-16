# Plot Output Structure

When you run the `canari_ml plot` command, it generates visualisation outputs that are stored in the `outputs/` directory. These outputs include plots and animations that allow you to visualise the model’s predictions and compare them against ground truth data.

---

## Overview

The `plot` subcommand will output visualisations comparing the prediction against ground truth. Currently only this plot is generated, but, in the future, this subcommand will expand out to more visualisations.

---

## Simplified Output Tree

```console
outputs/
└── demo_train/                     # Parent directory corresponding to the training run
    └── prediction/
        └── 1979-01-26/             # Prediction name
            └── 42/
                ├── predict_2025-09-16_12-23-04.log  # Plotting log file
                └── results/
                    └── ua700_comparison/          # Folder for comparison plots
                        └── 1979-01-26.mp4         # Animation comparing predictions
```

Where:

- `ua700_comparison/` contains the plots created for each of the forecast initialisation dates.

### Output Breakdown

`prediction/<forecast_date>/results/ua700_comparison/`

- The main output directory of the ua700 prediction vs ground truth plots.

`1979-01-26.mp4`

- Plot video comparing the ua700 variable for the forecast date 1979-01-26.
- If predictions have been made for multiple forecast initialisation dates, there will be multiple files here, corresponding to each date.

---

## Summary

After running the `canari_ml plot` command:

- The plots and comparison videos are stored under `outputs/<train.name>/prediction/<predict.name>/results/ua700_comparison/`.
