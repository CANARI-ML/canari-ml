import logging
import os

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from download_toolbox.interface import get_dataset_config_implementation
from icenet.plotting.forecast import ForecastPlotArgParser
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .cli import PlottingNumpyArgParser
from .utils import get_forecast_obs_data, high_res_rectangle

cm = mpl.colormaps

# Good plotting reference:
# https://kpegion.github.io/Pangeo-at-AOES/examples/multi-panel-cartopy.html


def plot_numpy_prediction(numpy_file: os.PathLike) -> None:
    """Plots direct forecast prediction output (numpy file) with interactive
    sliders.

    Args:
        numpy_file: Path to the numpy file containing forecast predictions.
    """
    prediction = np.load(numpy_file)

    # Get dimensions (time, height, width, leadtime)
    time_steps = prediction.shape[0]
    leadtimes = prediction.shape[3]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Initial set-up
    selected_time = 0
    selected_leadtime = 0

    pred_slice = prediction[selected_time, :, :, selected_leadtime]

    img = ax.imshow(pred_slice, cmap="viridis")
    ax.set_title(f"Time {selected_time + 1}, Leadtime {selected_leadtime + 1}")
    plt.colorbar(img, ax=ax)

    # Only want to create sliders if more than 1 element
    create_time_slider = time_steps > 1
    create_leadtime_slider = leadtimes > 1

    divider = make_axes_locatable(ax)
    if create_time_slider:
        time_slider_ax = divider.append_axes("bottom", "10%", pad=0.25)
        time_slider = Slider(
            ax=time_slider_ax,
            label="Time",
            valmin=1,
            valmax=time_steps,
            valinit=selected_time,
            valstep=np.linspace(1, time_steps, num=time_steps),
            orientation="horizontal",
        )
    if create_leadtime_slider:
        leadtime_slider_ax = divider.append_axes("bottom", "10%", pad=0.25)
        leadtime_slider = Slider(
            ax=leadtime_slider_ax,
            label="Leadtime",
            valmin=1,
            valmax=leadtimes,
            valinit=selected_leadtime,
            valstep=np.linspace(1, leadtimes, num=leadtimes),
            orientation="horizontal",
        )

    def update(val):
        """Update function for the sliders."""
        current_time = time_slider.val if create_time_slider else selected_time + 1
        current_leadtime = (
            leadtime_slider.val if create_leadtime_slider else selected_leadtime + 1
        )

        current_time = int(round(current_time))
        current_leadtime = int(round(current_leadtime))

        # Validate selections (clipping to available indices)
        if current_time < 1:
            current_time = 1
        elif current_time > time_steps:
            current_time = time_steps
        if current_leadtime < 1:
            current_leadtime = 1
        elif current_leadtime > leadtimes:
            current_leadtime = leadtimes

        # Extract new slice and update figure
        pred_slice = prediction[current_time - 1, :, :, current_leadtime - 1]
        img.set_data(pred_slice)
        ax.set_title(f"Time {current_time}, Leadtime {current_leadtime}")

        # Redraw figure
        fig.canvas.draw_idle()

    # Register update function with sliders
    if create_time_slider:
        time_slider.on_changed(update)
    if create_leadtime_slider:
        leadtime_slider.on_changed(update)

    # Button to reset to defaults
    reset_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_ax, "Reset")

    def reset(event):
        if create_time_slider:
            time_slider.reset()
        if create_leadtime_slider:
            leadtime_slider.reset()
        return

    reset_button.on_clicked(reset)

    plt.show()


def ua700_error_plot(
    fc_da: xr.DataArray,
    obs_da: xr.DataArray,
    obs_ds_config: object,
    output_path: os.PathLike,
    spatial_ref: object,
) -> None:
    """Plots ua700 forecast against ERA5 observations in EASE-Grid 2.0.

    Args:
        fc_da: Forecast data array to be plotted.
        obs_da: Observation data array for comparison.
        obs_ds_config: Configuration parameters for the observation dataset.
        output_path: Path where the plot will be saved.
        spatial_ref: Spatial reference configuration.
    """
    crs_wkt = spatial_ref["crs_wkt"]
    proj_crs = pyproj.CRS.from_wkt(crs_wkt)
    proj_ccrs = ccrs.Projection(proj_crs)
    proj_ccrs = ccrs.LambertAzimuthalEqualArea(0, 90)

    from canari_ml.preprocess.reproject import reproject_dataset_ease2

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 6),
        sharey=True,
        layout="constrained",
        subplot_kw={"projection": proj_ccrs},
    )
    axes = axes.flatten()
    ax1, ax2 = axes

    gridlines_kwargs = dict(
        draw_labels=True,
        dms=True,
        # x_inline=False,
        # y_inline=True,
        auto_inline=True,
        xlocs=range(-180, 180, 30),
        ylocs=range(0, 90, 10),
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
        crs=ccrs.PlateCarree(),
    )

    # Set extent to northern hemisphere - this should be fixed for canari-ml
    lon_min, lon_max = -180, 180
    lat_min, lat_max = 0, 90

    # Create a rectangular polygon (in PlateCarree coordinates)
    northern_region_path = mpath.Path(
        [
            [lon_min, lat_min],  # Bottom-left corner
            [lon_max, lat_min],  # Bottom-right corner
            [lon_max, lat_max],  # Top-right corner
            [lon_min, lat_max],  # Top-left corner
            [lon_min, lat_min],  # Close the loop
        ]
    )

    northern_region_path = high_res_rectangle(
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        target_crs=axes[0].projection,
        num_points=200,
    )

    # land = cartopy.feature.NaturalEarthFeature('physical', 'land', \
    #     scale="110m", edgecolor='k', facecolor=cfeature.COLORS['land'])
    for ax in axes.flat:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.coastlines(resolution="110m", linewidth=1, color="black")
        # ax.add_feature(land, linewidth=0.2, linestyle='--', edgecolor='k', alpha=1)
        ax.gridlines(**gridlines_kwargs)

        # Set the boundary on the axes without a further transform,
        # since the boundary is already in the axes' coordinate system.
        ax.set_boundary(northern_region_path)

        # ax.set_frame_on(False)  # Hide boundary frame, else drawing line by longitude boundary

    obs_da = reproject_dataset_ease2(obs_da, target_crs=proj_ccrs)
    ua_min, ua_max = int(obs_da.min().data), int(obs_da.max().data)
    ua_min, ua_max = -15, 25
    contour_level_step = 5

    plot_kwargs = dict(
        cmap="RdBu_r", extend="both", levels=range(ua_min, ua_max, contour_level_step)
    )

    contour_kwargs = dict(
        colors="black",
        add_labels=True,
        linewidths=0.3,
        levels=range(ua_min, ua_max, contour_level_step),
    )

    # TODO: Take time variable as an input
    # TODO: Really, a refactor pulling generalisable plotting components to a separate module
    time = 0
    im = fc_da.isel(time=time).plot.pcolormesh(
        ax=ax1, add_colorbar=False, **plot_kwargs
    )
    fc_da.isel(time=time).plot.contour(ax=ax1, **contour_kwargs)
    im = obs_da.isel(time=time).plot.pcolormesh(
        ax=ax2, add_colorbar=False, **plot_kwargs
    )
    obs_da.isel(time=time).plot.contour(ax=ax2, **contour_kwargs)

    tic = f"{pd.to_datetime(fc_da.isel(time=time).time.values).strftime(obs_ds_config.frequency.plot_format)}"
    tio = f"{pd.to_datetime(obs_da.isel(time=time).time.values).strftime(obs_ds_config.frequency.plot_format)}"
    # tic, tio = "", ""

    ax1.set_title(f"CANARI-ML Forecast\n{fc_da.long_name}\n{tic}")
    ax2.set_title(f"ERA5 Analysis (EASE-Grid 2.0)\n{obs_da.long_name}\n{tio}")

    # Draw the colorbar
    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation="horizontal",
        # fraction=1.0,
        # shrink=0.8,
        aspect=50,
        # pad=0.05,
        label=fc_da.units,
    )

    plt.suptitle("CANARI-ML Prediction against ERA5 observation")
    plt.show()


def plot_numpy():
    """CLI entrypoint to plot a direct numpy prediction output"""
    args = PlottingNumpyArgParser().parse_args()
    plot_numpy_prediction(args.numpy_file)


def plot_ua700_error():
    """
    Produces plot comparing ua700 forecast and ground truth.
    """
    ap = ForecastPlotArgParser()
    args = ap.parse_args()

    fc, obs, spatial_ref = get_forecast_obs_data(
        args.forecast_file, args.obs_dataset_config, args.forecast_date
    )
    ds_config = get_dataset_config_implementation(args.obs_dataset_config)

    logging.info("Plotting ua700 error")

    ua700_error_plot(
        fc_da=fc,
        obs_da=obs,
        obs_ds_config=ds_config,
        output_path=args.output_path,
        spatial_ref=spatial_ref,
    )
