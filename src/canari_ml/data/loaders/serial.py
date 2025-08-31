import datetime as dt
import logging
import os
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from pprint import pformat

import dask
import dask.array as da
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta
from icenet.data.loaders.base import DATE_FORMAT
from tqdm import tqdm
from zarr.convenience import consolidate_metadata

from canari_ml.data.loaders.base import CanariMLBaseDataLoader

logger = logging.getLogger(__name__)

# Speeds up matplotlib rendering a lot!
matplotlib.use("Agg")


class SerialLoader(CanariMLBaseDataLoader):
    """
    A loader that generates and loads data serially.

    This class extends `CanariMLBaseDataLoader` to provide functionality for
    generating and loading data sequentially. It supports generating data for
    multiple datasets (e.g., 'train', 'val', 'test') and can optionally produce
    plots for each sample. The generation process is configurable with various
    arguments, including batch size, number of workers, dry mode, and plot output.

    Args:
        *args: Variable length argument list.
        plot (optional): Whether to also output plots for each sample. Defaults to False.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        _masks (dict[str, xr.DataArray]): Dictionary of masks for each variable, loaded from configuration.
        _plot (bool): Flag indicating whether to produce plots during data generation.
    """
    def __init__(self, *args, plot=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._plot = plot

        self._masks = {
            var_name: xr.open_dataarray(mask_cfg["processed_files"][var_name][0])
            for var_name, mask_cfg in self._config["masks"].items()
        }

    def generate(self) -> None:
        """
        Initiate data generation process using the client.
        """
        self.client_generate(dates_override=self.dates_override, pickup=self.pickup)

    def client_generate(
        self,
        dates_override: dict | None = None,
        pickup: bool = False,
        client: object | None = None,
    ):
        """
        Generate data for multiple datasets sequentially.

        This method generates data for 'train', 'val', and 'test' datasets
        in sequence. It supports overriding dates using `dates_override` argument
        and can pick up an existing generation process using `pickup`.

        Args:
            dates_override (optional): Dates to override for each split.
                Should be a dictionary with keys 'train', 'val', and 'test',
                where the values are lists of dates. Defaults to None.
            pickup (optional): Whether to pick up an existing generation process.
                Defaults to False.
            client: Client object.
                    Defaults to None.

        Returns:
            None
        """
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
                logger.info(
                    "{} available {} dates".format(len(forecast_dates), dataset)
                )
                forecast_dates = forecast_dates.intersection(dates_override[dataset])
            forecast_dates = sorted(list(forecast_dates))

            output_dir = self.get_data_var_folder(dataset)
            zarr_path = os.path.join(output_dir, f"{dataset}.zarr")

            logger.info(
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

                logger.debug(f"Forecast dates:\n{pformat(forecast_dates)}")

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

                logger.info("Finished output {}".format(zarr_data))
                counts[dataset] += samples
                exec_times += gen_times
            else:
                counts[dataset] += len(forecast_dates)
                logger.warning("Skipping {} on pickup run".format(zarr_path))

        if len(exec_times) > 0:
            logger.info(
                "Average sample generation time: {}".format(np.average(exec_times))
            )
        self._write_dataset_config(counts)

    def generate_sample(self, date: object, prediction: bool = False, parallel=True) -> tuple[np.array, np.array, np.array]:
        """
        Generate a sample for the given date.

        This method generates a single data sample for the provided date using
        the configured variables and masks.

        Args:
            date (datetime.date): The date to generate a sample for.
            prediction (optional): Whether requesting a sample for predictions instead of targets.
                                   Defaults to False.
            parallel (optional): Whether to read the data from multiple `nc` filee in parallel.
                                 Defaults to True.

        Returns:
            A tuple containing the input features,
                output target, and sample weights for the generated sample.
        """
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

        logger.debug("VAR: {}".format(var_ds))
        var_ds = var_ds.transpose("yc", "xc", "time")

        trend_files = \
            [v for k, v in var_files.items()
             if k.endswith("linear_trend")]
        trend_ds = None

        if len(trend_files) > 0:
            trend_ds = xr.open_mfdataset(trend_files, **ds_kwargs)
            logger.debug("TREND: {}".format(trend_ds))
            trend_ds = trend_ds.transpose("yc", "xc", "time")

        args = [
            self._channels, self._dtype, self._loss_weight_days,
            self._meta_channels, self._missing_dates, self._lead_time,
            self.num_channels, self._shape, self._trend_steps, self._frequency_attr,
            self._masks, prediction
        ]

        var_ds.close()

        if prediction:
            x, base_ua700, y, sw = generate_sample(date, var_ds, var_files, trend_ds, *args)
            return x.compute(), base_ua700.compute(), y.compute(), sw.compute()
        else:
            x, y, sw = generate_sample(date, var_ds, var_files, trend_ds, *args)
            return x.compute(), y.compute(), sw.compute()


def plot_samples_grid(
    data_array, title_prefix, fname, titles=None, vmin=0, vmax=1, cmap="RdBu_r"
):
    """
    Plot samples in a grid.

    Args:
        data_array: 3D array (N, H, W), where N is the number of channels
        title_prefix: Prefix for figure title
        fname: Output file path (.jpg)
        titles (optional): List of strings to title each subplot
        cmap (optional): Matplotlib colormap
    """
    n_slices = data_array.shape[-1]
    n_cols = 5
    n_rows = int(np.ceil(n_slices / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4, n_rows * 4),
        constrained_layout=True
    )

    # Normalise axes format
    axes = np.atleast_2d(axes)

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
        else:
            ax.set_visible(False)

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", shrink=0.2, pad=0.05)

    fig.suptitle(title_prefix, fontsize=14)
    fig.savefig(fname, dpi=300)
    plt.close(fig)


def process_date(idx: int,
                 date: dt.date,
                 n_forecast_steps: int, # `lead_time` variable
                 var_ds: xr.Dataset,
                 var_files: dict[str, str],
                 trend_ds: xr.Dataset,
                 channels: dict[str, int],
                 meta_channels: list[str],
                 trend_steps: list[int] | int,
                 frequency_attr: str,
                 dry: bool,
                 plot: bool,
                 plot_dir: str,
                 args: tuple,
                ) -> tuple[np.array, np.array, np.array, float]:
    """
    Process a single date to generate samples and write them to the Zarr store.

    This function generates a sample for the given date using the provided datasets and
    configuration arguments. It writes the generated sample to the specified Zarr store
    if not running in dry mode. Optionally, it outputs plots of inputs, outputs, and
    sample weights for visualisation.

    Args:
        idx: Index of the current date.
        date: Date to generate samples for.
        n_forecast_steps: Number of `days/months/...` to forecast for.
        x_store: Zarr store array for input data.
        y_store: Zarr store array for output data.
        sw_store: Zarr store array for sample weights.
        var_ds: Dataset containing variable data.
        var_files: Dictionary of variable files with their corresponding paths.
        trend_ds: Dataset containing linear trend data (if any).
        channels: Dictionary mapping variable names to the number of channels.
        meta_channels: List of metadata channel names.
        trend_steps: Trend steps for linear trends (if applicable).
        frequency_attr: Attribute indicating the time frequency (e.g., "months" or "days").
        dry: Whether to run in dry mode. Default is False.
        plot: Whether to output plots for each sample. Default is False.
        plot_dir: Directory path for saving plots.
        args: Additional arguments required for generating samples.

    Returns:
        A tuple of:
            * x: inputs,
            * y: target,
            * sample_weights: sample weights
            * and the time taken to process the date in seconds.
    """
    start = time.time()
    # Generate sample for the date
    x, y, sample_weights = generate_sample(
        date, var_ds, var_files, trend_ds, *args
    )
    if not dry:
        x[da.isnan(x)] = 0.0

        # Output plots of inputs, outputs and sample weights
        if plot:
            x, y, sample_weights = dask.compute(
                x, y, sample_weights, optimize_graph=True
            )
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
                    x_titles.reverse()

            # Leadtime labels
            relative_attr = frequency_attr + "s"  # e.g. "months" or "days"
            _, _, forecast_steps_gen = get_date_indices(date, var_ds, n_forecast_steps, relative_attr)
            forecast_steps = list(forecast_steps_gen)
            lead_titles = [date_obj.strftime("%Y-%m-%d") for date_obj in forecast_steps]

            # lead_titles = [
            #     (date + relativedelta(**{relative_attr: i})).strftime("%Y-%m-%d")
            #     for i in range(y.shape[2])
            # ]

            # Plot grids with colorbars and labels
            # x has dims: (time*classes, height, width)
            # Reorder, since I've coded it for y which has time as the
            # last dimension.
            x_reordered = np.moveaxis(x, 0, -1) #(height, width, time*classes)
            plot_samples_grid(
                x_reordered, f"x - {date}",
                os.path.join(plot_dir, f"x_{idx}_{date}_grid.jpg"),
                titles=x_titles,
                vmin=0,
                vmax=1,
            )
            # y has dims: (output_classes, height, width, leadtime)
            # Where, output_classes is just `ua700` right now
            plot_samples_grid(
                y[0, :, :, :], f"y - {date}",
                os.path.join(plot_dir, f"y_{idx}_{date}_grid.jpg"),
                titles=lead_titles,
                vmin=-0.5,
                vmax=0.5,
            )
            # sample_weights has dims: (output_classes, height, width, leadtime)
            plot_samples_grid(
                sample_weights[0, :, :, :], f"sample_weights - {date}",
                os.path.join(plot_dir, f"sw_{idx}_{date}_grid.jpg"),
                titles=lead_titles,
                vmin=0,
                vmax=1,
            )

    x, y, sample_weights = dask.compute(x, y, sample_weights)

    end = time.time()
    duration = end - start
    logger.info(f"Time taken to produce {date}: {duration:.4f} seconds")
    return x, y, sample_weights, duration


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
        batch_size (optional): Batch size for processing.
                               Defaults to 32.
        workers (optional): Number of worker processes for parallel processing.
                            Defaults to 4.
        dry (optional): Whether to run in dry mode.
                        Defaults to False.
        plot (optional): Whether to also output plots for each sample.
                         Defaults to False.

    Returns:
        Paths to the output Zarr dataset, the count of processed
            dates, and a list of time taken for each date.
    """
    count = 0
    times = [0.0]*len(dates)

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
        parallel=False,
    )

    for k, v in var_files.items():
        if k not in meta_channels and not k.endswith("linear_trend"):
            print("k, v:", k, v)

    trend_files = [v for k, v in var_files.items() if k.endswith("linear_trend")]
    trend_ds = None

    if len(trend_files):
        trend_ds = xr.open_mfdataset(trend_files, **ds_kwargs)
        trend_ds = trend_ds.transpose("yc", "xc", "time")

    # Directory to save plots
    plot_dir = os.path.join(os.path.dirname(path), "plots")
    if plot:
        os.makedirs(plot_dir, exist_ok=True)

    # Only predicting ua700 in model output
    out_channels = 1

    empty_x = da.zeros((len(dates), num_channels, *shape), dtype=dtype, chunks=(batch_size, num_channels, *shape))
    empty_y = da.zeros((len(dates), out_channels, *shape, lead_time), dtype=dtype, chunks=(batch_size, out_channels, *shape, lead_time))
    empty_sw = da.zeros((len(dates), out_channels, *shape, lead_time), dtype=dtype, chunks=(batch_size, out_channels, *shape, lead_time))

    # Pre-allocate empty Zarr dataset
    ## Zarr does not like saving dates as is with dtype=object.
    time_coord = np.array(dates, dtype="datetime64[ns]")
    ds = xr.Dataset(
        {
            "x": (["time", "channels", "yc", "xc"], empty_x),
            "y": (["time", "channel", "yc", "xc", "lead_time"], empty_y),
            "sample_weights": (["time", "channel", "yc", "xc", "lead_time"], empty_sw),
        },
        coords={"time": time_coord},
    )

    # # Chunk dataset
    # chunks=(batch_size, num_channels, *shape, )
    # ds["x"] = ds["x"].chunk(chunks)
    # chunks=(batch_size, out_channels, *shape, lead_time)
    # ds["y"] = ds["y"].chunk(chunks)
    # ds["sample_weights"] = ds["sample_weights"].chunk(chunks)

    # Write empty Zarr ds to initialise
    ds.to_zarr(path, mode="w", consolidated=False)

    def worker(idx_date):
        idx, date = idx_date

        with xr.open_mfdataset(
            [
                v
                for k, v in var_files.items()
                if k not in meta_channels and not k.endswith("linear_trend")
            ],
            **ds_kwargs,
            engine="h5netcdf", # Found default netcdf4 engine buggy
        ) as var_ds:

            x, y, sw, duration = process_date(
                                            idx,
                                            date,
                                            lead_time,
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

        return idx, x, y, sw, duration


    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in tqdm(executor.map(worker, enumerate(dates)), total=len(dates)):
            idx, x, y, sample_weights, duration = result

            # # Write results back to Zarr in batch
            # for idx, x, y, sample_weights, duration in results:
            ds_region = xr.Dataset(
                {
                    "x": (["time", "channels", "yc", "xc"], x[np.newaxis]),
                    "y": (["time", "channel", "yc", "xc", "lead_time"], y[np.newaxis]),
                    "sample_weights": (["time", "channel", "yc", "xc", "lead_time"], sample_weights[np.newaxis]),
                },
                coords={"time": [time_coord[idx]]},
            )

            ds_region.to_zarr(path, mode="a", region={"time": slice(idx, idx+1)}, consolidated=False)
            # ds_region.to_zarr(path, mode="a", consolidated=False)

            times[idx] = duration
            count += 1


    # # Option provided by zarr logger:
    # # 3. Explicitly setting consolidated=True, to raise an error in this case instead of falling back to try reading non-consolidated metadata.
    # #   xr.open_zarr(path).to_zarr(path, mode="a", consolidated=True)
    # xr.open_zarr(path).to_zarr(path, mode="a", consolidated=True)

    # # Create final consolidated metadata using zarr instead
    consolidate_metadata(path)

    return path, count, times


def get_date_indices(forecast_date: dt.datetime,
        var_ds: xr.Dataset,
        n_forecast_steps: int,
        relative_attr: str,
        ) -> (int, list[int], Generator):
    """
    Compute the indices and dates need as inputs and outputs to the forecast model.

    Given a forecast initialisation date, the input dataset, the number of steps to
    forecast for, and a relative time attribute (e.g., 'months', 'days').

    Args:
        forecast_date: The initialisation date for the forecast.
        var_ds: xarray Dataset.
        n_forecast_steps: Number of forecast steps (lead times) to generate.
        relative_attr: The time attribute for stepping forward (e.g., "months", "days").

    Returns:
        forecast_base_idx: The index of the forecast init date in `var_ds.time`.
        forecast_idxs: List of indices for each forecast step in `var_ds.time`.
        forecast_steps: Generator yielding the dates for each forecast step.
    """
    forecast_base_idx = list(var_ds.time.values).index(pd.Timestamp(forecast_date))
    forecast_idxs = [forecast_base_idx + n for n in range(0, n_forecast_steps)]

    def forecast_steps():
        for leadtime_idx in range(n_forecast_steps):
            forecast_step = forecast_date + relativedelta(**{relative_attr: leadtime_idx})
            yield forecast_step

    return forecast_base_idx, forecast_idxs, forecast_steps()


def get_channel_idxs(var_name: str,
    forecast_base_idx: int,
    num_channels: int,
    trend_steps: int | list[int]
    ) -> list[int]:
    """
    Compute the time indices for input channels for a given variable.

    Determine which time indices to use for a variable's input channels,
    depending on whether the variable is a linear trend or a lagged variable.

    Args:
        var_name: Name of the variable. If it ends with "linear_trend", trend logic is used.
        forecast_base_idx: Index of the forecast initialisation date in the time dimension.
        num_channels: Number of channels to generate for this variable.
        trend_steps: Steps to use for trend channels.
            If list: use these as offsets from the base index.
            If int: use a range from 0 to `num_channels-1` as offsets.

    Returns:
        List of indices corresponding to the time dimension for each channel.
    """
    if var_name.endswith("linear_trend"):
        if type(trend_steps) is list:
            channel_idxs = [forecast_base_idx + n for n in trend_steps]
        else:
            channel_idxs = [forecast_base_idx + n for n in range(0, num_channels)]
    # If we're not a trend, we're a lag channel looking back historically from the
    # initialisation date
    else:
        channel_idxs = [forecast_base_idx - n for n in range(1, num_channels + 1)]

    return channel_idxs


def generate_sample(
    forecast_date: object,
    var_ds: object,
    var_files: dict,
    trend_ds: object,
    channels: dict,
    dtype: object,
    loss_weight_days: bool,
    meta_channels: list,
    missing_dates: list,
    n_forecast_steps: int,
    num_channels: int,
    shape: object,
    trend_steps: object,
    frequency_attr: str,
    masks: object,
    prediction: bool = False,
):
    """
    Generate a sample for train/val/prediction.

    This function creates input features (x), targets (y), and sample weights based on
    the given parameters.

    Args:
        forecast_date: The forecast initialisation date.
        var_ds: The input xarray dataset containing variables like ua700_abs, siconca, etc.
        var_files: Map of meta variable names to their corresponding file paths.
        trend_ds: The xarray dataset containing linear trends.
        channels: Map of variable name to number of channels(excluding meta).
        dtype: The data type used for the input features, targets, and sample weights.
        loss_weight_days: If True, apply temporal weighting for loss calculation.
        meta_channels: Meta channel names to include in the input features.
        missing_dates: Dates with missing data.
        n_forecast_steps: The number of forecast steps in target (target leadtime).
        num_channels: The total number of channels (input features).
        shape: The spatial shape of the dataset.
        trend_steps: The step(s) for linear trends. Can be a single integer or a list of integers.
        frequency_attr: The time frequency attribute, e.g., 'DAY' for daily data.
        masks: Map of mask names and their corresponding DataArrays.
        prediction (optional): If True, generate a sample for prediction; otherwise, generate a training sample.
                               Defaults to False.

    Returns:
        x: Input features with shape (num_channels, *shape).
        y: Targets with shape (1, *shape, n_forecast_steps).
        sample_weights: Sample weights with shape (1, *shape, n_forecast_steps).
    """
    # DAYS/MONTHS/YEARS
    relative_attr = "{}s".format(frequency_attr)

    masks["hemisphere"] = masks["hemisphere"].astype(bool)

    # Prepare data sample
    # To become array of shape (*raw_data_shape, n_forecast_steps)
    forecast_base_idx, forecast_idxs, forecast_steps_gen = get_date_indices(
        forecast_date, var_ds, n_forecast_steps, relative_attr
    )

    n_output_channels = 1 # Just ua700 in output prediction

    y = da.zeros((n_output_channels, *shape, n_forecast_steps), dtype=dtype)
    sample_weights = da.zeros((n_output_channels, *shape, n_forecast_steps), dtype=dtype)

    # Get ua700 for the day/month before the forecast initialisation date
    base_ua700 = var_ds.ua700_abs.isel(time=forecast_base_idx - 1)
    if not prediction:
        try:
            sample_output = var_ds.ua700_abs.isel(time=forecast_idxs).transpose("yc", "xc", "time")

            # Add time dimension to end
            base_ua700_expanded = base_ua700.expand_dims(time=sample_output.time)

            # Set model target values to be delta to the day/month before the
            # forecast initialisation date
            sample_output = sample_output - base_ua700_expanded
        except KeyError as sic_ex:
            logger.exception(
                "Issue selecting data for non-prediction sample, "
                "please review ua700 ground-truth: dates {}".format(forecast_idxs)
            )
            raise RuntimeError(sic_ex)
        y[0, :, :, :] = sample_output
        if "hemisphere" in masks:
            y_mask = da.stack(
                [masks["hemisphere"].data for _ in range(0, n_output_channels)], axis=0
            )
            y_mask = da.stack([y_mask], axis=-1)
            y = da.ma.where(y_mask, 0.0, y)

    # Masked recomposition of output
    # Loop through the generator with dates we're predicting for
    for leadtime_idx, forecast_step in enumerate(forecast_steps_gen):
        if any([forecast_step == missing_date for missing_date in missing_dates]):
            sample_weight = da.zeros(shape, dtype)
        else:
            # No masking when sample_weight = 1
            sample_weight = np.ones(shape, dtype=dtype)
            if "weighted_regions" in masks:
                sample_weight = masks["weighted_regions"].data
            if "hemisphere" in masks:
                # Zero loss across the mask hemisphere
                # (i.e., outside of northern hemisphere)
                hemisphere_mask = masks["hemisphere"].data
                sample_weight[hemisphere_mask] = 0.0
            sample_weight = sample_weight.astype(dtype)

            # We can pick up nans, which messes up training
            sample_weight[da.isnan(y[0, ..., leadtime_idx])] = 0.0

        sample_weights[0, :, :, leadtime_idx] = sample_weight

    # INPUT FEATURES
    x = da.zeros((num_channels, *shape), dtype=dtype)
    v1, v2 = 0, 0

    for var_name, num_channels in channels.items():
        if var_name in meta_channels:
            continue

        v2 += num_channels

        channel_idxs = get_channel_idxs(
            var_name, forecast_base_idx, num_channels, trend_steps
        )
        channel_ds = trend_ds if var_name.endswith("linear_trend") else var_ds

        channel_data = []
        for idx in channel_idxs:
            try:
                data = getattr(channel_ds, var_name).isel(time=idx)
                if var_name.startswith("siconca"):
                    data = da.ma.where(masks["land"], 0.0, data)
                channel_data.append(data)

                # logger.info("NANs: {} = {} in {}-{}".format(forecast_date, int(da.isnan(data).sum()), var_name, idx))
            except KeyError as e:
                logger.warning(
                    "KeyError detected on channel construction for {} - {}: {}".format(
                        var_name, idx, e
                    )
                )
                channel_data.append(da.zeros(shape))

        x[v1:v2, :, :] = da.from_array(channel_data)#.transpose([0, 1, 2])
        v1 += num_channels

    for var_name in meta_channels:
        if channels[var_name] > 1:
            raise RuntimeError(
                "{} meta variable cannot have more than one channel".format(var_name)
            )

        with xr.open_dataarray(var_files[var_name]) as meta_ds:
            if var_name in ["sin", "cos"]:
                ref_date = "2012-{}-{}".format(forecast_date.month, forecast_date.day)
                trig_val = meta_ds.sel(time=ref_date).to_numpy()
                x[v1, :, :] = da.broadcast_to([trig_val], shape)
            else:
                x[v1, :, :] = da.array(meta_ds.to_numpy())
            v1 += channels[var_name]

    # TODO: we have unwarranted nans which need fixing, probably from broken spatial infilling
    nan_mask_x, nan_mask_y, nan_mask_sw = (
        da.isnan(x),
        da.isnan(y),
        da.isnan(sample_weights),
    )
    if nan_mask_x.sum() + nan_mask_y.sum() + nan_mask_sw.sum() > 0:
        logger.warning(
            "NANs: {} in input, {} in output, {} in weights".format(
                int(nan_mask_x.sum()), int(nan_mask_y.sum()), int(nan_mask_sw.sum())
            )
        )
        x[nan_mask_x] = 0
        sample_weights[nan_mask_sw] = 0
        y[nan_mask_y] = 0

    if prediction:
        return x, base_ua700, y, sample_weights
    else:
        return x, y, sample_weights
