# Postprocess

After generating raw numpy predictions, we can postprocess these raw predictions using the `canari_ml postprocess` command to create netCDF files.

We can use the same prediction config file from the previous step to generate these files. It will create an aggregate netCDF file with `ua700_mean` and `ua700_stddev` for all ensembles.

```console
canari_ml postprocess -cd configs/predict/custom_train -cn 1979-01-26_and_plot
```
