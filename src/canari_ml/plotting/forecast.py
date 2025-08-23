import datetime as dt
import logging
import os
from pathlib import Path

import cartopy.crs as ccrs
import imageio_ffmpeg as ffmpeg
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from download_toolbox.interface import get_dataset_config_implementation
from hydra.core.hydra_config import HydraConfig
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import DictConfig

from canari_ml.preprocess.reproject import ease2_reference_grid_setup, reproject_dataset

from .utils import get_axes, get_forecast_obs_data

cm = mpl.colormaps

# Set Matplotlib's ffmpeg executable path to the one from imageio_ffmpeg
ffmpeg_path = ffmpeg.get_ffmpeg_exe()
plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path

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
    crs_wkt: str,
    show_plot: bool = False,
) -> None:
    """Plots ua700 forecast against ERA5 observations in EASE-Grid 2.0.

    Args:
        fc_da: Forecast data array to be plotted.
        obs_da: Observation data array for comparison.
        obs_ds_config: Configuration parameters for the observation dataset.
        output_path: Path where the plot will be saved.
        crs_wkt: WKT string of the projection to be used for plotting.
        show_plot: Whether to show the plot or save animation to file.
                   Defaults to False.
    """
    proj_crs = pyproj.CRS.from_wkt(crs_wkt)
    proj_ccrs = ccrs.Projection(proj_crs)
    proj_ccrs = ccrs.LambertAzimuthalEqualArea(0, 90)
    proj_epsg = f"EPSG:{proj_ccrs.to_epsg()}"

    # Get figure and geoaxes.
    fig_kwargs = dict(
        nrows=1,
        ncols=2,
        sharey=True,
        subplot_kw={"projection": proj_ccrs},
    )
    gridlines_kwargs = dict(
        xlocs=range(-180, 180, 30),
        ylocs=range(0, 90, 10),
    )
    fig, axes = get_axes(fig_kwargs=fig_kwargs, gridlines_kwargs=gridlines_kwargs)
    ax1, ax2 = axes

    shape = fc_da.isel(time=0).shape

    # Plot comparison of observation with ground truth.
    grid, mask = ease2_reference_grid_setup(shape=shape, target_crs=proj_epsg)
    obs_da = reproject_dataset(obs_da, grid, mask, target_crs=proj_epsg)
    ua_min, ua_max = int(obs_da.min().data), int(obs_da.max().data)
    # ua_min, ua_max = -15, 25
    contour_level_step = 7

    # Set common plotting parameters
    plot_kwargs = dict(
        cmap="RdBu_r", extend="both", levels=range(ua_min, ua_max, contour_level_step)
    )
    contour_kwargs = dict(
        colors="black",
        add_labels=True,
        linewidths=0.3,
        levels=range(ua_min, ua_max, contour_level_step),
        zorder=1,
    )

    # Initial plots
    ## First axis
    im1 = fc_da.isel(time=0).plot.pcolormesh(ax=ax1, add_colorbar=False, **plot_kwargs)
    im2 = fc_da.isel(time=0).plot.contour(ax=ax1, **contour_kwargs)
    ## Second axis
    im3 = obs_da.isel(time=0).plot.pcolormesh(ax=ax2, add_colorbar=False, **plot_kwargs)
    im4 = fc_da.isel(time=0).plot.contour(ax=ax2, **contour_kwargs)
    artists = [im1, im2, im3, im4]

    cbar = fig.colorbar(
        im1,
        ax=axes,
        orientation="horizontal",
        # fraction=1.0,
        # shrink=0.8,
        aspect=50,
        # pad=0.05,
        label=fc_da.units,
    )

    plt.suptitle("CANARI-ML Prediction against ERA5 observation")

   # Create time labels for slider (if show plot instead of save animation flag enabled)
    time_labels = [
        pd.to_datetime(t.item()).strftime(obs_ds_config.frequency.plot_format)
        for t in fc_da.time
    ]

    def update(frame):
        for artist in artists:
            artist.remove()
        artists.clear()

        # Plot data
        im1 = fc_da.isel(time=frame).plot.pcolormesh(ax=ax1, add_colorbar=False, **plot_kwargs)
        im2 = fc_da.isel(time=frame).plot.contour(ax=ax1, **contour_kwargs)
        im3 = obs_da.isel(time=frame).plot.pcolormesh(ax=ax2, add_colorbar=False, **plot_kwargs)
        im4 = obs_da.isel(time=frame).plot.contour(ax=ax2, **contour_kwargs)
        artists.extend([im1, im2, im3, im4])

        # Updating titles from dataset - cleaner approach.
        tic = pd.to_datetime(fc_da.isel(time=frame).time.values).strftime(obs_ds_config.frequency.plot_format)
        tio = pd.to_datetime(obs_da.isel(time=frame).time.values).strftime(obs_ds_config.frequency.plot_format)
        # tic, tio = "", ""
        ax1.set_title(f"CANARI-ML Forecast\n{fc_da.long_name}\n{tic}")
        ax2.set_title(f"ERA5 Analysis (EASE-Grid 2.0)\n{obs_da.long_name}\n{tio}")

    # Disabling constrained after I've added plots & colorbar to prevent jumping around in
    # successive frames.
    fig.set_constrained_layout(False)

    if show_plot:
        slider_start, slider_end = 0.2, 0.6
        slider_y = 0.03  # vertical position
        tick_height = 0.01  # height of the tick in figure coordinates

        ax_slider = fig.add_axes([slider_start, 0.03, slider_end, 0.05])
        time_slider = Slider(
            ax_slider,
            "Forecast date",
            0,
            len(fc_da.time) - 1,
            valinit=0,
            valstep=1,
            valfmt="%d",
        )

        # Set label to first time label immediately
        # else, will show 0 when first frame is plotted
        time_slider.valtext.set_text(time_labels[0])

        # Add tickmarks so I know start/end of forecast along slider
        num_ticks = min(len(fc_da.time), 10)
        tick_indices = np.linspace(0, len(fc_da.time) - 1, num_ticks, dtype=int)

        for idx in tick_indices:
            rel_x = idx / (len(fc_da.time) - 1)  # normalised position [0, 1]
            tick_x = slider_start + rel_x * slider_end  # match slider's position (0.2 to 0.8)

            # Emulate tick links so I know selection points
            fig.lines.append(plt.Line2D(
                [tick_x, tick_x],
                [slider_y, slider_y + tick_height],
                transform=fig.transFigure,
                color="black",
                linewidth=1
            ))

        def update_slider_label(val):
            index = int(val)
            time_slider.valtext.set_text(time_labels[index])
            update(index)

        time_slider.on_changed(update_slider_label)
        update(0)
        plt.show()
    else:
        anim = FuncAnimation(fig, update, frames=len(fc_da.time), interval=500)

        output_path = os.path.join("plots", "ua700_comparison.mp4") \
            if not output_path else output_path
        logging.info(f"Saving to {output_path}")

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        writer = animation.FFMpegWriter(
            fps=2,
            metadata={"artist": "CANARI-ML"},
            extra_args=['-pix_fmt', 'yuv420p']
        )

        anim.save(output_path, writer=writer, dpi=300)

        plt.close(fig)


def plot_ua700_error(cfg: DictConfig):
    """
    Produces plot comparing ua700 forecast and ground truth.
    """

    run_dir = HydraConfig.get().runtime.output_dir
    nc_path = cfg.paths.postprocess.netcdf_path
    nc_file = os.path.join(nc_path, cfg.postprocess.netcdf.name)
    # dates = [dt.date(*[int(v) for v in s.split("-")]) for s in cfg.predict.dates]
    out_video_path = os.path.join(run_dir, f"{cfg.plot.name}")

    source_data_config_file = cfg.paths.download.config_file

    for date in cfg.predict.dates:
        fc, obs, spatial_ref = get_forecast_obs_data(
            Path(nc_file), source_data_config_file, date
        )

        ds_config = get_dataset_config_implementation(source_data_config_file)
        logging.info("Plotting ua700 results")

        out_video_file = os.path.join(out_video_path, f"{date}.mp4")
        ua700_error_plot(
            fc_da=fc,
            obs_da=obs,
            obs_ds_config=ds_config,
            output_path=Path(out_video_file),
            crs_wkt=spatial_ref["crs_wkt"],
            show_plot=cfg.plot.show,
        )
