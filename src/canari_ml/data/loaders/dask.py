import datetime as dt
import logging
import os

import numpy as np
import xarray as xr
from icenet.data.loaders.base import DATE_FORMAT
from icenet.data.loaders.dask import (
    DaskBaseDataLoader,
)

from canari_ml.data.loaders.serial import SerialLoader, generate_and_write


class DaskMultiWorkerLoader(DaskBaseDataLoader, SerialLoader):
    def __init__(self, *args, futures_per_worker: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._masks = {
            var_name: xr.open_dataarray(mask_cfg["processed_files"][var_name][0])
            for var_name, mask_cfg in self._config["masks"].items()
        }

        self._futures = futures_per_worker

    def client_generate(
        self, client: object, dates_override: object = None, pickup: bool = False
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

        masks = client.scatter(self._masks, broadcast=True)

        for dataset in splits:
            batch_number = 0
            futures = []

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

                    fut = client.submit(
                        generate_and_write,
                        tf_path.format(batch_number),
                        self.get_sample_files(),
                        dates,
                        args,
                        dry=self._dry,
                    )
                    futures.append(fut)

                    # Use this to limit the future list, to avoid crashing the
                    # distributed scheduler / workers (task list gets too big!)
                    if len(futures) >= self._workers * self._futures:
                        for tf_data, samples, gen_times in client.gather(futures):
                            logging.info("Finished output {}".format(tf_data))
                            counts[dataset] += samples
                            exec_times += gen_times
                        futures = []

                    # tf_data, samples, times = generate_and_write(
                    #    tf_path.format(batch_number), args, dry=self._dry)
                else:
                    counts[dataset] += len(dates)
                    logging.warning(
                        "Skipping {} on pickup run".format(tf_path.format(batch_number))
                    )

                batch_number += 1

            # Hoover up remaining futures
            for tf_data, samples, gen_times in client.gather(futures):
                logging.info("Finished output {}".format(tf_data))
                counts[dataset] += samples
                exec_times += gen_times

        if len(exec_times) > 0:
            logging.info(
                "Average sample generation time: {}".format(np.average(exec_times))
            )
        self._write_dataset_config(counts)
