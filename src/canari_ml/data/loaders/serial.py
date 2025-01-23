import datetime as dt
import logging
import os
import time

import dask
import dask.array as da
import numpy as np
import xarray as xr
import zarr
from icenet.data.loaders.base import DATE_FORMAT, IceNetBaseDataLoader
from icenet.data.loaders.dask import (
    generate_sample,
)


class SerialLoader(IceNetBaseDataLoader):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._masks = {
            var_name: xr.open_dataarray(mask_cfg["processed_files"][var_name][0])
            for var_name, mask_cfg in self._config["masks"].items()
        }

    def generate(self) -> None:
        self.client_generate(dates_override=self.dates_override, pickup=self.pickup)

    def client_generate(
        self,
        dates_override: object = None,
        pickup: bool = False,
        client: object | None = None,
    ):
        # TODO: for each set, validate every variable has an appropriate file
        #  in the configuration arrays, otherwise drop the forecast date
        splits = ("train", "val", "test")

        if dates_override and type(dates_override) is dict:
            for split in splits:
                assert (
                    split in dates_override.keys()
                    and type(dates_override[split]) is list
                ), "{} needs to be list in dates_override".format(split)
        elif dates_override:
            raise RuntimeError("dates_override needs to be a dict if supplied")

        counts = {el: 0 for el in splits}
        exec_times = []

        masks = self._masks

        # Loop through ('train', 'val', 'test')
        for dataset in splits:
            # Make sure we have a unique set of forecast_dates
            forecast_dates = set(
                [
                    dt.datetime.strptime(s, DATE_FORMAT).date()
                    for identity in self._config["sources"].keys()
                    for s in self._config["sources"][identity]["splits"][dataset]
                ]
            )

            if dates_override:
                logging.info(
                    "{} available {} dates".format(len(forecast_dates), dataset)
                )
                forecast_dates = forecast_dates.intersection(dates_override[dataset])
            forecast_dates = sorted(list(forecast_dates))

            output_dir = self.get_data_var_folder(dataset)
            zarr_path = os.path.join(output_dir, f"{dataset}.zarr")

            logging.info(
                "{} {} dates to process, generating cache data.".format(
                    len(forecast_dates), dataset
                )
            )

            if not pickup or (pickup and not os.path.exists(zarr_path)):
                args = [
                    self._channels,
                    self._dtype,
                    self._loss_weight_days,
                    self._meta_channels,
                    self._missing_dates,
                    self._lead_time,
                    self.num_channels,
                    self._shape,
                    self._trend_steps,
                    self._frequency_attr,
                    masks,
                    False,
                ]

                zarr_data, samples, gen_times = generate_and_write(
                    zarr_path,
                    self.get_sample_files(),
                    forecast_dates,
                    args,
                    batch_size=self._output_batch_size,
                    dry=self._dry,
                )

                logging.info("Finished output {}".format(zarr_data))
                counts[dataset] += samples
                exec_times += gen_times
            else:
                counts[dataset] += len(forecast_dates)
                logging.warning("Skipping {} on pickup run".format(zarr_path))

        if len(exec_times) > 0:
            logging.info(
                "Average sample generation time: {}".format(np.average(exec_times))
            )
        self._write_dataset_config(counts)

    def generate_sample(self) -> None:
        pass


def generate_and_write(
    path: str,
    var_files: dict[str, str],
    dates: list[dt.date],
    args: tuple,
    batch_size: int = 32,
    dry: bool = False,
) -> tuple[str, int, list[float]]:
    """
    Generate and write Zarr dataset.

    Args:
        path: Path to the output Zarr dataset.
        var_files: Dictionary of variable files with their corresponding paths.
        dates: List of dates to generate samples for.
        args: Method arguments.
        dry (optional): Whether to run in dry mode. Defaults to False.

    Returns:
        Tuple containing the path to the output Zarr dataset, the count of processed
            dates, and a list of time taken for each date.
    """
    count = 0
    times = []

    (
        channels,
        dtype,
        loss_weight_days,
        meta_channels,
        missing_dates,
        n_forecast_days,
        num_channels,
        shape,
        trend_steps,
        frequency_attr,
        masks,
        prediction,
    ) = args

    ds_kwargs = dict(
        chunks=dict(time=1, yc=shape[0], xc=shape[1]),
        drop_variables=["month", "plev", "realization"],
        parallel=True,
    )

    var_ds = xr.open_mfdataset(
        [
            v
            for k, v in var_files.items()
            if k not in meta_channels and not k.endswith("linear_trend")
        ],
        **ds_kwargs,
    )
    var_ds = var_ds.transpose("yc", "xc", "time")

    trend_files = [v for k, v in var_files.items() if k.endswith("linear_trend")]
    trend_ds = None

    if len(trend_files):
        trend_ds = xr.open_mfdataset(trend_files, **ds_kwargs)
        trend_ds = trend_ds.transpose("yc", "xc", "time")

    # Prepare Zarr store
    with zarr.open(path, mode="w") as store:
        # Prepare arrays for x, y, and sample_weights
        x_store = store.create_dataset(
            "x",
            shape=(len(dates), *shape, num_channels),
            dtype=dtype,
            chunks=(batch_size, *shape, num_channels),
            compressor=zarr.Blosc(cname="zstd", clevel=3),
        )
        y_store = store.create_dataset(
            "y",
            shape=(len(dates), *shape, n_forecast_days, 1),
            dtype=dtype,
            chunks=(batch_size, *shape, n_forecast_days, 1),
            compressor=zarr.Blosc(cname="zstd", clevel=3),
        )
        sw_store = store.create_dataset(
            "sample_weights",
            shape=(len(dates), *shape, n_forecast_days, 1),
            dtype=dtype,
            chunks=(batch_size, *shape, n_forecast_days, 1),
            compressor=zarr.Blosc(cname="zstd", clevel=3),
        )

        for idx, date in enumerate(dates):
            start = time.time()

            x, y, sample_weights = generate_sample(
                date, var_ds, var_files, trend_ds, *args
            )
            if not dry:
                x[da.isnan(x)] = 0.0

                x, y, sample_weights = dask.compute(
                    x, y, sample_weights, optimize_graph=True
                )
                print(x.shape, y.shape, sample_weights.shape)

                # Write to Zarr store
                x_store[idx] = x
                y_store[idx] = y
                sw_store[idx] = sample_weights

            count += 1

            end = time.time()
            times.append(end - start)
            logging.debug(f"Time taken to produce {date}: {times[-1]:.4f} seconds")

    return path, count, times
