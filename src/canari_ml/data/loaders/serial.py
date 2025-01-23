import datetime as dt
import logging
import os
import time
from pprint import pformat

import dask
import dask.array as da
import numpy as np
import tensorflow as tf
import xarray as xr
from icenet.data.loaders.base import DATE_FORMAT, IceNetBaseDataLoader
from icenet.data.loaders.dask import (
    generate_sample,
)
from icenet.data.loaders.utils import write_tfrecord


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

        def batch(batch_dates, num):
            i = 0
            while i < len(batch_dates):
                yield batch_dates[i : i + num]
                i += num

        masks = self._masks

        for dataset in splits:
            batch_number = 0

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
            tf_path = os.path.join(output_dir, "{:08}.tfrecord")

            logging.info(
                "{} {} dates to process, generating cache data.".format(
                    len(forecast_dates), dataset
                )
            )

            for dates in batch(forecast_dates, self._output_batch_size):
                if not pickup or (
                    pickup and not os.path.exists(tf_path.format(batch_number))
                ):
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

                    tf_data, samples, gen_times = generate_and_write(
                        tf_path.format(batch_number),
                        self.get_sample_files(),
                        dates,
                        args,
                        dry=self._dry,
                    )

                    logging.info("Finished output {}".format(tf_data))
                    counts[dataset] += samples
                    exec_times += gen_times
                else:
                    counts[dataset] += len(dates)
                    logging.warning(
                        "Skipping {} on pickup run".format(tf_path.format(batch_number))
                    )

                batch_number += 1

        if len(exec_times) > 0:
            logging.info(
                "Average sample generation time: {}".format(np.average(exec_times))
            )
        self._write_dataset_config(counts)

    def generate_sample(self, date: object, prediction: bool = False, parallel=True):
        ds_kwargs = dict(
            chunks=dict(time=1, yc=self._shape[0], xc=self._shape[1]),
            drop_variables=["month", "plev", "level", "realization"],
            parallel=parallel,
        )
        var_files = self.get_sample_files()

        var_ds = xr.open_mfdataset(
            [
                v
                for k, v in var_files.items()
                if k not in self._meta_channels and not k.endswith("linear_trend")
            ],
            **ds_kwargs,
        )

        logging.debug("VAR: {}".format(pformat(var_ds)))
        var_ds = var_ds.transpose("yc", "xc", "time")

        trend_files = [v for k, v in var_files.items() if k.endswith("linear_trend")]
        trend_ds = None

        if len(trend_files) > 0:
            trend_ds = xr.open_mfdataset(trend_files, **ds_kwargs)
            logging.debug("TREND: {}".format(pformat(trend_ds)))
            trend_ds = trend_ds.transpose("yc", "xc", "time")

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
            self._masks,
            prediction,
        ]

        x, y, sw = generate_sample(date, var_ds, var_files, trend_ds, *args)
        return x.compute(), y.compute(), sw.compute()


def generate_and_write(
    path: str, var_files: object, dates: object, args: tuple, dry: bool = False
):
    count = 0
    times = []

    # TODO: refactor, this is very smelly - with new data throughput args
    #  will always be the same
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

    with tf.io.TFRecordWriter(path) as writer:
        for date in dates:
            start = time.time()

            x, y, sample_weights = generate_sample(
                date, var_ds, var_files, trend_ds, *args
            )
            if not dry:
                x[da.isnan(x)] = 0.0

                x, y, sample_weights = dask.compute(
                    x, y, sample_weights, optimize_graph=True
                )
                write_tfrecord(writer, x, y, sample_weights)
            count += 1

            end = time.time()
            times.append(end - start)
            logging.debug("Time taken to produce {}: {}".format(date, times[-1]))

    return path, count, times
