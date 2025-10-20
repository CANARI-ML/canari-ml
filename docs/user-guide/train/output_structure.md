# Training Output Structure

When you run the `canari_ml train` command, all outputs are written to the `outputs/` directory by default. This structure contains model checkpoints, logs, and symlinks to relevant preprocessed data used during training. The directory is organised to support both training reproducibility and downstream evaluation.

## Overview

The training pipeline performs the following:

- Loads training/validation datasets from cached Zarr files.
- Uses associated normalisation parameters for input processing.
- Trains the model and logs metrics (e.g., loss, RMSE).
- Saves model checkpoints and logs under a uniquely identified run folder.
- Symlinks key preprocessing directories to preserve provenance and reproducibility.

---

## Simplified Output Tree

```bash
outputs/
└── demo_train/                     # Output folder for a training run (named via config or CLI)
    └── training/                   # Container for training experiment
        ├── 42/                     # Specific training ensemble seed
        │   ├── checkpoints/        # Model checkpoint directory
        │   │   ├── epoch=1-val_rmse=0.2113.ckpt  # Best checkpoint from run, including epoch and validation metric in the filename
        │   │   └── last.ckpt                        # Latest checkpoint at end of training, to be used for resuming in future
        │   └── train_2025-09-16_10-06-30.log        # Log file with timestamped training output
        │
        ├── cache_dir -> ../../../preprocessed_data/train_demo_dataset/03_cache_demo_dataset/   # Symlink to cached Zarr dataset
        └── normalisation_dir -> ../../../preprocessed_data/preprocessed/02_normalised_demo_dataset/era5/   # Symlink to normalisation parameters, to be used when generating prediction dataset
```

Where:

- `demo_train/` is the name of the training run, specified via the training config (`train.name`) or CLI flag.

- `42/` is the random seed used to generate the training model.

- `cache_dir` and `normalisation_dir` symlinks which point back to the preprocessed data used for training.

### Output Breakdown

`training/42/`

- This is the primary directory for a single training run. Its name (`42/`) will vary depending on seed specified, and there will be multiple directories corresponding to each seed if running multiple ensembles.

`checkpoints/`

- Contains model checkpoints saved during training.

- Naming convention includes validation RMSE, e.g.:
    - `epoch=1-val_rmse=0.2113.ckpt` – a checkpoint saved after epoch 1 with corresponding validation RMSE.

- `last.ckpt` – the most recent checkpoint, useful for resuming training (not currently coded) or evaluation.

`train_<timestamp>.log`

- Training log file with the date and time the run began.

- Includes configuration summary, epoch-level training and validation metrics, and any warnings/errors.

#### Symlinks

`cache_dir -> preprocessed_data/train_demo_dataset/03_cache_demo_dataset/`

- Points to the final cached Zarr dataset used during training.
- Ensures exact data provenance and reproducibility.
- Contains `train/`, `val/`, `test/` Zarr groups and metadata JSON (see Preprocessing Output Structure).

`normalisation_dir -> preprocessed_data/preprocessed/02_normalised_demo_dataset/era5/`

- Symlink to the directory containing normalisation parameters.
- Will be used for prediction dataset normalisation.

## Summary

After running the `canari_ml train` command:

- All model outputs are placed under the `outputs/` directory.
- Training run logs and checkpoints are stored in a uniquely identified subdirectory (`outputs/<name>/training/<seed>/`).
- Symlinks to key preprocessed data components (`cache_dir` and `normalisation_dir`) are included to maintain clear data lineage.
- This structure is designed for:
    - Reproducibility: Every training run keeps track of the exact data and parameters used.
    - Modularity: Outputs can be cleanly separated and referenced for evaluation or further training.
