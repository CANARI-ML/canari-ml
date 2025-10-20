# Generate Predictions

After creating a trained model, we can generate a dataset for prediction, utilising the same normalisation as the dataset used for training. This is done using the `canari_ml preprocess` command, which will generate a dataset in the `preprocessed_data/` directory.

## Structure

Since we can generate multiple predictions for one trained model, we can create a directory with the same name as the trained model, and then create the hydra yaml config file in that directory to specify the parameters of the prediction. i.e.:

```bash
configs/predict/custom_train/1979-01-26.yaml
```

This is just a recommendation, and you can use the directory structure to hold your config files that you prefer.

This step will generate raw numpy prediction files which must be post-processed to denormalise the predictions, and obtain an equivalent to the ground truth `ua700`.
