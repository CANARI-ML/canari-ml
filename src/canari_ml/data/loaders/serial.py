import datetime as dt
import logging
import os
import time

import dask
import dask.array as da
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from concurrent.futures import ProcessPoolExecutor, as_completed
from dateutil.relativedelta import relativedelta
from icenet.data.loaders.base import DATE_FORMAT, IceNetBaseDataLoader
from joblib import Parallel, delayed

# Speeds up matplotlib rendering a lot!
matplotlib.use("Agg")


class SerialLoader(IceNetBaseDataLoader):
    def __init__(self, *args, plot=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._plot = plot

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

                logging.debug(f"Forecast dates:\n{pformat(forecast_dates)}")

                zarr_data, samples, gen_times = generate_and_write(
                    zarr_path,
                    self.get_sample_files(),
                    forecast_dates,
                    args,
                    batch_size=self._output_batch_size,
                    workers=self._workers,
                    dry=self._dry,
                    plot=self._plot,
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

    def generate_sample(self, date: object, prediction: bool = False, parallel=True) -> None:
        ds_kwargs = dict(
            chunks=dict(time=1, yc=self._shape[0], xc=self._shape[1]),
            drop_variables=["month", "plev", "level", "realization"],
            parallel=parallel,
            engine="h5netcdf",
        )
        var_files = self.get_sample_files()

        var_ds = xr.open_mfdataset([
            v for k, v in var_files.items()
            if k not in self._meta_channels and not k.endswith("linear_trend")
        ], **ds_kwargs)

        logging.debug("VAR: {}".format(var_ds))
        var_ds = var_ds.transpose("yc", "xc", "time")

        trend_files = \
            [v for k, v in var_files.items()
             if k.endswith("linear_trend")]
        trend_ds = None

        if len(trend_files) > 0:
            trend_ds = xr.open_mfdataset(trend_files, **ds_kwargs)
            logging.debug("TREND: {}".format(trend_ds))
            trend_ds = trend_ds.transpose("yc", "xc", "time")

        args = [
            self._channels, self._dtype, self._loss_weight_days,
            self._meta_channels, self._missing_dates, self._lead_time,
            self.num_channels, self._shape, self._trend_steps, self._frequency_attr,
            self._masks, prediction
        ]

        x, y, sw = generate_sample(date, var_ds, var_files, trend_ds, *args)
        return x.compute(), y.compute(), sw.compute()


def plot_samples_grid(data_array, title_prefix, fname, titles=None, cmap="RdBu_r"):
    """
    Plot samples in a grid.

    Args:
        data_array: 3D array (H, W, N)
        title_prefix: Prefix for figure title
        fname: Output file path (.jpg)
        titles: List of strings to title each subplot
        cmap: Matplotlib colormap
    """
    n_slices = data_array.shape[-1]
    n_cols = 5
    n_rows = int(np.ceil(n_slices / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4, n_rows * 4),
        constrained_layout=True
    )

    # Normalize axes format
    axes = np.atleast_2d(axes)

    vmin = np.nanmin(data_array)
    vmax = np.nanmax(data_array)

    im = None
    for i in range(n_rows * n_cols):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        ax.axis("off")

        if i < n_slices:
            im = ax.imshow(
                data_array[:, :, i],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            if titles:
                ax.set_title(titles[i], fontsize=8)

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", shrink=0.2, pad=0.05)

    fig.suptitle(title_prefix, fontsize=14)
    fig.savefig(fname, dpi=300)
    plt.close(fig)


def process_date(idx,
                 date,
                 x_store,
                 y_store,
                 sw_store,
                 var_ds,
                 var_files,
                 trend_ds,
                 channels,
                 meta_channels,
                 trend_steps,
                 frequency_attr,
                 dry,
                 plot,
                 plot_dir,
                 args
                ):
    start = time.time()
    # Generate sample for the date
    x, y, sample_weights = generate_sample(
        date, var_ds, var_files, trend_ds, *args
    )
    if not dry:
        x[da.isnan(x)] = 0.0

        x, y, sample_weights = dask.compute(
            x, y, sample_weights, optimize_graph=True
        )

        # Write to Zarr store
        x_store[idx] = x
        y_store[idx] = y
        sw_store[idx] = sample_weights

        # Output plots of inputs, outputs and sample weights
        if plot:
            # Build channel names from config
            x_titles = []
            for var_name, num_ch in channels.items():
                if var_name in meta_channels:
                    for _ in range(num_ch):
                        x_titles.append(var_name)
                elif var_name.endswith("linear_trend"):
                    for step in trend_steps if isinstance(trend_steps, list) else range(num_ch):
                        x_titles.append(f"{var_name}_t{step}")
                else:
                    for lag in range(num_ch):
                        x_titles.append(f"{var_name}_lag{lag}")

            # Leadtime labels
            relative_attr = frequency_attr + "s"  # e.g. "months" or "days"
            lead_titles = [
                (date + relativedelta(**{relative_attr: i})).strftime("%Y-%m-%d")
                for i in range(y.shape[2])
            ]

            # Plot grids with colorbars and labels
            plot_samples_grid(
                x, f"x - {date}",
                os.path.join(plot_dir, f"x_{idx}_{date}_grid.jpg"),
                titles=x_titles
            )
            plot_samples_grid(
                y[:, :, :, 0], f"y - {date}",
                os.path.join(plot_dir, f"y_{idx}_{date}_grid.jpg"),
                titles=lead_titles
            )
            plot_samples_grid(
                sample_weights[:, :, :, 0], f"sample_weights - {date}",
                os.path.join(plot_dir, f"sw_{idx}_{date}_grid.jpg"),
                titles=lead_titles
            )

    end = time.time()
    duration = end - start
    logging.info(f"Time taken to produce {date}: {duration:.4f} seconds")
    return idx, date, duration

def generate_and_write(
    path: str,
    var_files: dict[str, str],
    dates: list[dt.date],
    args: tuple,
    batch_size: int = 32,
    workers: int = 4,
    dry: bool = False,
    plot: bool = False,
) -> tuple[str, int, list[float]]:
    """
    Generate and write Zarr dataset.

    Args:
        path: Path to the output Zarr dataset.
        var_files: Dictionary of variable files with their corresponding paths.
        dates: List of dates to generate samples for.
        args: Method arguments.
        dry (optional): Whether to run in dry mode. Defaults to False.
        plot (optional): Whether to also output plots for each sample. Defaults to False.

    Returns:
        Paths to the output Zarr dataset, the count of processed
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
        lead_time,
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


    for k, v in var_files.items():
        if k not in meta_channels and not k.endswith("linear_trend"):
            print("k, v:", k, v)

    var_ds = xr.open_mfdataset(
        [
            v
            for k, v in var_files.items()
            if k not in meta_channels and not k.endswith("linear_trend")
        ],
        **ds_kwargs,
        engine="h5netcdf", # Found default netcdf4 engine buggy
    )

    trend_files = [v for k, v in var_files.items() if k.endswith("linear_trend")]
    trend_ds = None

    if len(trend_files):
        trend_ds = xr.open_mfdataset(trend_files, **ds_kwargs)
        trend_ds = trend_ds.transpose("yc", "xc", "time")

    # Directory to save plots
    plot_dir = os.path.join(os.path.dirname(path), "plots")
    if plot:
        os.makedirs(plot_dir, exist_ok=True)

    # Prepare Zarr store
    with zarr.open(path, mode="w") as store:
        # Prepare arrays for x, y, and sample_weights
        x_store = store.create_dataset(
            "x",
            shape=(len(dates), *shape, num_channels),
            dtype=dtype,
            chunks=(batch_size, *shape, num_channels),
            # compressor=zarr.Blosc(cname="zstd", clevel=3),
        )
        y_store = store.create_dataset(
            "y",
            shape=(len(dates), *shape, lead_time, 1),
            dtype=dtype,
            chunks=(batch_size, *shape, lead_time, 1),
            # compressor=zarr.Blosc(cname="zstd", clevel=3),
        )
        sw_store = store.create_dataset(
            "sample_weights",
            shape=(len(dates), *shape, lead_time, 1),
            dtype=dtype,
            chunks=(batch_size, *shape, lead_time, 1),
            # compressor=zarr.Blosc(cname="zstd", clevel=3),
        )
        # Use self._workers (assumes you're in a class method)
        times = []
        count = 0

        if workers == 1:
            for idx, date in enumerate(dates):
                idx, date, duration = process_date(
                                                    idx,
                                                    date,
                                                    x_store,
                                                    y_store,
                                                    sw_store,
                                                    var_ds,
                                                    var_files,
                                                    trend_ds,
                                                    channels,
                                                    meta_channels,
                                                    trend_steps,
                                                    frequency_attr,
                                                    dry,
                                                    plot,
                                                    plot_dir,
                                                    args
                                                  )
                times.append(duration)
                count += 1
        else:
            # TODO: Switch to Dask based approach - though, workers clash
            # with pytorch num_workers in dataloader.
            results = Parallel(n_jobs=workers, backend="loky", timeout=9999, verbose=51)(
                delayed(process_date)(
                                        idx,
                                        date,
                                        x_store,
                                        y_store,
                                        sw_store,
                                        var_ds,
                                        var_files,
                                        trend_ds,
                                        channels,
                                        meta_channels,
                                        trend_steps,
                                        frequency_attr,
                                        dry,
                                        plot,
                                        plot_dir,
                                        args
                ) for idx, date in enumerate(dates)
            )
            for idx, date, duration in results:
                times.append(duration)
                count += 1
            # with ProcessPoolExecutor(max_workers=workers) as executor:
            #     futures = {
            #         executor.submit(process_date,
            #                         idx,
            #                         date,
            #                         x_store,
            #                         y_store,
            #                         sw_store,
            #                         var_ds,
            #                         var_files,
            #                         trend_ds,
            #                         channels,
            #                         meta_channels,
            #                         trend_steps,
            #                         frequency_attr,
            #                         dry,
            #                         plot,
            #                         plot_dir,
            #                         args
            #                         ): (idx, date)
            #         for idx, date in enumerate(dates)
            #     }

            #     for future in as_completed(futures):
            #         idx, date, duration = future.result()
            #         times.append(duration)
            #         count += 1

    return path, count, times


def generate_sample(
    forecast_date: object,
    var_ds: object,
    var_files: object,
    trend_ds: object,
    channels: object,
    dtype: object,
    loss_weight_days: bool,
    meta_channels: object,
    missing_dates: object,
    n_forecast_steps: int,
    num_channels: int,
    shape: object,
    trend_steps: object,
    frequency_attr: str,
    masks: object,
    prediction: bool = False,
):
    # DAYS/MONTHS/YEARS
    relative_attr = "{}s".format(frequency_attr)

    # Prepare data sample
    # To become array of shape (*raw_data_shape, n_forecast_steps)
    forecast_base_idx = list(var_ds.time.values).index(pd.Timestamp(forecast_date))
    forecast_idxs = [forecast_base_idx + n for n in range(0, n_forecast_steps)]

    y = da.zeros((*shape, n_forecast_steps, 1), dtype=dtype)
    sample_weights = da.zeros((*shape, n_forecast_steps, 1), dtype=dtype)

    if not prediction:
        try:
            sample_output = var_ds.ua700_abs.isel(time=forecast_idxs)
            sample_output = sample_output.transpose("yc", "xc", "time") # New
        except KeyError as sic_ex:
            logging.exception(
                "Issue selecting data for non-prediction sample, "
                "please review ua700 ground-truth: dates {}".format(forecast_idxs)
            )
            raise RuntimeError(sic_ex)
        y[:, :, :, 0] = sample_output
        if "hemisphere" in masks:
            y_mask = da.stack(
                [masks["hemisphere"].data for _ in range(0, n_forecast_steps)], axis=-1
            )
            y_mask = da.stack([y_mask], axis=-1)
            y = da.ma.where(y_mask, 0.0, y)

    # Masked recomposition of output
    for leadtime_idx in range(n_forecast_steps):
        forecast_step = forecast_date + relativedelta(**{relative_attr: leadtime_idx})

        if any([forecast_step == missing_date for missing_date in missing_dates]):
            sample_weight = da.zeros(shape, dtype)
        else:
            # No masking when sample_weight = 1
            sample_weight = np.ones(shape, dtype)
            if "hemisphere" in masks:
                # Zero loss across the mask hemisphere
                # (i.e., outside of northern hemisphere)
                hemisphere_mask = masks["hemisphere"].data
                sample_weight[hemisphere_mask] = 0.0
            sample_weight = sample_weight.astype(dtype)

            # We can pick up nans, which messes up training
            sample_weight[da.isnan(y[..., leadtime_idx, 0])] = 0.0

        sample_weights[:, :, leadtime_idx, 0] = sample_weight

    # INPUT FEATURES
    x = da.zeros((*shape, num_channels), dtype=dtype)
    v1, v2 = 0, 0

    for var_name, num_channels in channels.items():
        if var_name in meta_channels:
            continue

        v2 += num_channels

        if var_name.endswith("linear_trend"):
            channel_ds = trend_ds
            if type(trend_steps) is list:
                channel_idxs = [forecast_base_idx + n for n in trend_steps]
            else:
                channel_idxs = [forecast_base_idx + n for n in range(0, num_channels)]
        # If we're not a trend, we're a lag channel looking back historically from the initialisation date
        else:
            channel_ds = var_ds
            channel_idxs = [forecast_base_idx - n for n in range(1, num_channels + 1)]

        channel_data = []
        for idx in channel_idxs:
            try:
                data = getattr(channel_ds, var_name).isel(time=idx)
                if var_name.startswith("siconca"):
                    data = da.ma.where(masks["land"], 0.0, data)
                channel_data.append(data)

                # logging.info("NANs: {} = {} in {}-{}".format(forecast_date, int(da.isnan(data).sum()), var_name, idx))
            except KeyError as e:
                logging.warning(
                    "KeyError detected on channel construction for {} - {}: {}".format(
                        var_name, idx, e
                    )
                )
                channel_data.append(da.zeros(shape))

        x[:, :, v1:v2] = da.from_array(channel_data).transpose([1, 2, 0])
        v1 += num_channels

    for var_name in meta_channels:
        if channels[var_name] > 1:
            raise RuntimeError(
                "{} meta variable cannot have more than one channel".format(var_name)
            )

        meta_ds = xr.open_dataarray(var_files[var_name])

        if var_name in ["sin", "cos"]:
            ref_date = "2012-{}-{}".format(forecast_date.month, forecast_date.day)
            trig_val = meta_ds.sel(time=ref_date).to_numpy()
            x[:, :, v1] = da.broadcast_to([trig_val], shape)
        else:
            x[:, :, v1] = da.array(meta_ds.to_numpy())
        v1 += channels[var_name]

    # TODO: we have unwarranted nans which need fixing, probably from broken spatial infilling
    nan_mask_x, nan_mask_y, nan_mask_sw = (
        da.isnan(x),
        da.isnan(y),
        da.isnan(sample_weights),
    )
    if nan_mask_x.sum() + nan_mask_y.sum() + nan_mask_sw.sum() > 0:
        logging.warning(
            "NANs: {} in input, {} in output, {} in weights".format(
                int(nan_mask_x.sum()), int(nan_mask_y.sum()), int(nan_mask_sw.sum())
            )
        )
        x[nan_mask_x] = 0
        sample_weights[nan_mask_sw] = 0
        y[nan_mask_y] = 0

    return x, y, sample_weights
