# Plot

After generating the netCDFs, we can create comparison plots of the prediction against the ground truth data. This is done using the `canari_ml plot` command.

We can use the same prediction config file from the previous two steps to generate these files.

```console
canari_ml plot -cd configs/predict/custom_train -cn 1979-01-26_and_plot
```
