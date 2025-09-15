# Override Defaults via Command-Line

The `canari_ml train` command uses a structured configuration system, where you can override default settings using CLI arguments or YAML config files. 

## Override Train Parameters

- `train.seed`: Random seed for reproducibility.
- `train.epochs`: Number of training epochs.
- `train.workers`: Number of CPU/GPU workers for data loading and training.
- `train.batch_size`: Batch size for training.

## Override Model Parameters

You can also override model-specific parameters:

``` console
canari_ml train model.network.filter_size=5 model.litmodule.criterion.learning_rate=0.001
```

- `model.network.filter_size`: Filter size in the UNet architecture.
- `model.litmodule.criterion.learning_rate`: Learning rate for the optimiser.

## Override Callbacks

- `callbacks.early_stopping.patience`: Number of epochs to wait before applying early stopping.
- `callbacks.model_checkpoint.monitor`: Metric to monitor for saving checkpoints.

# Examples

## Basic Override Example
``` console
canari_ml train train.dataset=preprocessed_data/train_demo_dataset/03_cache_demo_dataset/cached.DAY.north.json train.name=demo_train train.epochs=2
```

## Advanced Override Example
``` console
canari_ml train train.dataset=preprocessed_data/train_demo_dataset/03_cache_demo_dataset/cached.DAY.north.json train.name=demo_train train.seed=42 train.epochs=20 train.workers=8 train.batch_size=16
```

## Override Callbacks

To modify callbacks like early stopping and checkpoint monitoring, add the following overrides to the above command:

``` console
callbacks.early_stopping.patience=5 callbacks.model_checkpoint.monitor=val_rmse
```
