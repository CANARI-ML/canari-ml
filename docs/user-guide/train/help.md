# Configuration Descriptions

Below is the default configuration structure for a training run:

| Name                  | Type      | Default Value           | Description                                                                 |
|-----------------------|-----------|-------------------------|-----------------------------------------------------------------------------|
| dataset.name          | str       | `era5`                 | Name of the dataset being trained on                                       |
| seed                  | int       | 42                      | Random seed for reproducibility                                            |
| epochs                | int       | 50                       | Number of training epochs                                                  |
| workers               | int       | 4                        | Number of data loading workers                                             |
| batch_size            | int       | 4                        | Batch size for training                                                    |
| shuffling             | bool      | true                     | Whether to shuffle the dataset during training                             |
| wandb_group           | str       | `unet`                  | Group name for Weights & Biases (W&B) tracking                              |
| wandb_project         | str       | `CANARI`                | Project name for W&B tracking                                              |
| verbose               | bool      | true                     | Whether to enable verbose training output                                  |

### Model Configuration

| Name                    | Type          | Default Value           | Description                                                                 |
|-------------------------|---------------|-------------------------|-----------------------------------------------------------------------------|
| model_name              | str            | `unet`                  | Name of the model architecture                                            |
| network.filter_size     | int            | 3                       | Size of convolutional filter kernel                                       |
| network.n_filters_factor| float          | 1.0                     | Factor for scaling the number of filters                                 |
| n_output_classes        | int            | 1                       | Number of output classes (e.g., predict ua at a specific pressure level) |

### Training Configuration

| Name                    | Type          | Default Value           | Description                                                                 |
|-------------------------|---------------|-------------------------|-----------------------------------------------------------------------------|
| precision               | str            | `16-mixed`              | Mixed precision training mode                                             |
| max_epochs              | int            | 50                       | Maximum number of epochs to train                                        |
| log_every_n_steps       | int            | 5                        | Log every N steps during training                                         |

### Callbacks Configuration

#### ModelCheckpoint
| Name                    | Type          | Default Value           | Description                                                                 |
|-------------------------|---------------|-------------------------|-----------------------------------------------------------------------------|
| dirpath                 | str            | `outputs/${train.name}/training/` | Directory to save checkpoints                                             |
| filename                | str            | `epoch={epoch}-${callbacks.model_checkpoint.monitor}={${callbacks.model_checkpoint.monitor}:.4f}` | Naming pattern for checkpoints                                              |

#### EarlyStopping
| Name                    | Type          | Default Value           | Description                                                                 |
|-------------------------|---------------|-------------------------|-----------------------------------------------------------------------------|
| monitor                 | str            | `val_rmse`              | Metric to monitor for early stopping                                      |
| patience                | int            | 10                       | Number of epochs to wait before early stopping                            |
