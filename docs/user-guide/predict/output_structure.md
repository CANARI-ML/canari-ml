# Prediction Output Structure

When you run the `canari_ml predict` command, the output is written to the `outputs/` directory under the corresponding training run folder. The prediction outputs are organised by forecast date, model run, and include logs, raw predictions, and a reference to the input dataset used for inference.

---

## Overview

The prediction pipeline performs the following steps:

- Loads the trained model checkpoint from a specified training run.
- Loads the corresponding prediction dataset (preprocessed but not yet seen by the model).
- Generate predictions from the trained model.
- Saves raw predictions in `.npy` format, with one file per forecast start date.
- Logs the prediction generation and symlinks back to the dataset used for traceability.

---

## Simplified Output Tree

```bash
outputs/
└── demo_train/                      # Parent directory corresponding to the training run
    └── prediction/
        └── 1979-01-26/              # Prediction name defined in the config
            ├── 42/                  # Seed for this prediction (e.g. from training)
            │   ├── raw_predictions/
            │   │   └── 1979_01_26.npy  # NumPy array of raw model outputs for this forecast
            │   └── predict_2025-09-16_11-08-52.log  # Log file with timestamp of prediction run
            └── cache_dir -> ../../../../preprocessed_data/predict_1976_example/03_cache_1976_example/
```

Where:

- `demo_train/` is the name of the original training run the model was trained under.
- `1979-01-26/` is the name of the prediction run, specified via the training config (`predict.name`) or CLI flag.
- `42/` is the seed number (Matches the training seed).
- Symlinks are created to the cached input dataset (`cache_dir`) used during prediction.

### Output Breakdown

`prediction/<forecast_date>/`

- Each forecast date is organised into its own directory under `prediction/`.

`42/`

- A single prediction run corresponding to a trained model. This is the same as the seed used for the trained model.

`raw_predictions/`

- Contains .npy files of the model's raw output.
- File naming format: `YYYY_MM_DD.npy`, where the date corresponds to the forecast initialisation date.

`predict_<timestamp>.log`

- Log file for the prediction run.
- Includes configuration used, model checkpoint, input paths, and prediction status.

#### Symlinks

`cache_dir -> preprocessed_data/predict_1976_example/03_cache_1976_example/`

- Points to the prediction dataset used during inference.
- Ensures that the same inputs can be used for postprocessing or re-running prediction.

---

## Summary

After running `canari_ml predict`:

- Prediction results are stored under `outputs/<train.name>/prediction/<predict.name>/`.
- All prediction data, logs, and references are grouped by the forecast start date.
- Each run includes:
    - A `.npy` file with the raw predictions.
    - A symlink back to the input dataset (`cache_dir`) used for inference.
    - A detailed prediction log for traceability and debugging.
