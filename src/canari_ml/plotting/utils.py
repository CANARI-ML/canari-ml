import logging
import os

import cartopy.crs as ccrs
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from dateutil.relativedelta import relativedelta
from download_toolbox.dataset import DatasetConfig
from download_toolbox.interface import Frequency, get_dataset_config_implementation
from icenet.plotting.utils import broadcast_forecast
from download_toolbox.utils import get_implementation
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import transform


def get_forecast_data(
    forecast_file: os.PathLike, forecast_date: str, stddev: bool = False
) -> tuple[xr.DataArray, dict]:
    """
    Get forecast data for a specific date and optional standard deviation.

    Args:
        forecast_file: Path to a NetCDF (.nc) file containing forecast data.
        forecast_date: Initialisation date of the forecast in string format.
        stddev: Whether to return standard deviation instead of mean values.
            Defaults to False.

    Returns:
        A tuple containing two elements:
            - The forecast dataarray (either ua700_mean or ua700_stddev)
            - Spatial reference attributes from the dataset
                (created by rioxarray)
    """
    logging.info("Opening forecast {} for date {}".format(forecast_file, forecast_date))
    forecast_date = pd.to_datetime(forecast_date)
    forecast_ds = xr.open_dataset(forecast_file, decode_coords="all")
    forecast_ds = forecast_ds.sel(time=slice(forecast_date, forecast_date))

    return (
        forecast_ds.ua700_mean if not stddev else forecast_ds.ua700_stddev,
        forecast_ds["spatial_ref"].attrs,
    )


def filter_ds_by_obs(
    ds: object, obs_da: object, forecast_date: str, frequency: Frequency = Frequency.DAY
) -> xr.Dataset:
    """
    Filter a dataset based on observational data for a specific forecast date.

    Args:
        ds: Input forecast dataset.
        obs_da: Observational data array to use for filtering.
        forecast_date: Initialisation date of the forecast in string format.
        frequency: Time frequency of the observational dataset.
            Defaults to `download_toolbox.interface.Frequency.DAY`.

    Returns:
        Filtered forecast dataset compatible with observational data range.
    """
    forecast_date = pd.to_datetime(forecast_date)
    delta_attribute = "{}s".format(frequency.attribute)
    (start_date, end_date) = (
        forecast_date,
        forecast_date + relativedelta(**{delta_attribute: int(ds.leadtime.max()-1)}),
    )

    if len(obs_da.time) < len(ds.leadtime):
        if len(obs_da.time) < 1:
            raise RuntimeError(
                "No observational data available between {} and {}".format(
                    start_date.strftime("%D"), end_date.strftime("%D")
                )
            )

        logging.warning(
            "Observational data not available for full range of "
            "forecast lead times: obs {}-{} vs fc {}-{}".format(
                obs_da.time.to_series()[0].strftime(frequency.date_format),
                obs_da.time.to_series()[-1].strftime(frequency.date_format),
                start_date.strftime(frequency.date_format),
                end_date.strftime(frequency.date_format),
            )
        )

        (start_date, end_date) = (
            obs_da.time.to_series()[0],
            obs_da.time.to_series()[-1],
        )

    # TODO: Bug where it will error since leadtime is +1 what its expecting
    # We broadcast to get a nicely compatible dataset for plotting
    return broadcast_forecast(
        start_date=start_date,
        end_date=end_date,
        dataset=ds,
        frequency=frequency,
        # target=None, # netCDF path for saving
    )


def get_forecast_obs_data(
    forecast_file: os.PathLike,
    obs_ds_config: os.PathLike,
    forecast_date: str,
    stddev: bool = False,
) -> tuple[xr.DataArray, xr.DataArray, dict]:
    """
    Get both forecast and observational data for a specific date.

    This function retrieves forecast data and corresponding observational
    data for a given forecast file, dataset configuration, and initialisation
    date. It returns the filtered forecast data, observational data, and spatial
    reference attributes.

    Args:
        forecast_file: Path to a NetCDF (.nc) file containing forecast data.
        obs_ds_config: Path to JSON config file for the observational dataset.
            Will by default be under `data/era5/dataset_config.month.hemi.{hemisphere}.json`.
        forecast_date: Initialisation date of the forecast in string format.
        stddev: Whether to return standard deviation instead of mean values.
            Defaults to False.

    Returns:
        A tuple containing three elements:
            - forecast_da: Filtered forecast data array.
            - obs_da: Observational data array.
            - spatial_ref: Spatial reference attributes from the dataset
                (created by rioxarray)
    """
    forecast_da, spatial_ref = get_forecast_data(forecast_file, forecast_date, stddev)
    ds_config = get_dataset_config_implementation(obs_ds_config)
    obs_ds = ds_config.get_dataset(var_names=["ua700"])
    obs_ds = obs_ds.sel(
        # This time slice of observation does not match filter_ds_by_obs() function
        time=slice(
            pd.to_datetime(forecast_date),
            pd.to_datetime(forecast_date)
            + relativedelta(
                **{
                    "{}s".format(ds_config.frequency.attribute): int(
                        forecast_da.leadtime.max()
                    )
                }
            ),
        )
    )
    # Don't want to apply a mask for CANARI-ML forecasts
    # masks = get_implementation(xr.open_dataset(forecast_file).attrs["canari_ml_mask_implementation"])(ds_config)
    forecast_da = filter_ds_by_obs(
        forecast_da, obs_ds, forecast_date, ds_config.frequency
    )
    return forecast_da, obs_ds.ua700, spatial_ref  # , masks


def high_res_rectangle(
    lon_min: int | float,
    lon_max: int | float,
    lat_min: int | float,
    lat_max: int | float,
    target_crs=None,
    num_points: int = 100,
    source_crs: ccrs.CRS = ccrs.PlateCarree(),
) -> mpath.Path:
    """
    Create a high-resolution lat/lon "rectangle" by subdividing the corners into
    smaller segments.

    Creates a smooth polygon boundary around a rectangular region defined by
    longitude and latitude limits. It subdivides each edge into smaller
    segments for higher resolution, projects the coordinates using specified CRS,
    and returns the boundary as an matplotlib Path object for plotting.
    The result can be used for creating custom gridlines or region boundaries in
    cartographic visualisations.

    Args:
        lon_min: The minimum longitude (left edge).
        lon_max: The maximum longitude (right edge).
        lat_min: The minimum latitude (bottom edge).
        lat_max: The maximum latitude (top edge).
        target_crs (optional): Target coordinate reference system for projection.
            Defaults to None.
        num_points (optional): Number of points to interpolate along each side of
            the rectangle. Defaults to 100.
        source_crs (ccrs.CRS, optional): Source coordinate reference system for input
            data. Defaults to ccrs.PlateCarree().
    Returns:
        A high-resolution projected lat/lon bounded Path for matplotlib custom boundary.
    """
    # Generate evenly spaced points along the four sides of the rectangle
    lon = np.linspace(lon_min, lon_max, num_points)
    lat = np.linspace(lat_min, lat_max, num_points)

    # Bottom side (from lon_min to lon_max, at lat_min)
    bottom = np.column_stack([lon, np.full_like(lon, lat_min)])
    # Top side (from lon_min to lon_max, at lat_max)
    top = np.column_stack([lon, np.full_like(lon, lat_max)])
    # Left side (from lat_min to lat_max, at lon_min)
    left = np.column_stack([np.full_like(lat, lon_min), lat])
    # Right side (from lat_min to lat_max, at lon_max)
    right = np.column_stack([np.full_like(lat, lon_max), lat])

    # Combine all sides into a single list of points and close the polygon
    rectangle_points = np.vstack([bottom, right[1:], top[::-1], left[1:][::-1]])

    # Use Shapely so I can tidy up the result
    polygon = Polygon(rectangle_points)

    # Ensure the polygon is oriented correctly and valid
    polygon = orient(polygon)

    # Now, fix any topology issues in the projected polygon.
    # Fix small invalidities (e.g. self-intersections).
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    # Reproject polygon before tidy-up so there are no reprojection errors
    # creeping in before I tidy up the result.
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

    def proj_func(x, y, z=None):
        return transformer.transform(x, y)

    projected_polygon = transform(proj_func, polygon)

    # Fix any topology issues in the projected polygon.
    # Else, can get non-manifold shape which won't show the gridline labels
    # correctly
    if not projected_polygon.is_valid:
        projected_polygon = projected_polygon.buffer(0)

    # Simplify polygon.
    projected_polygon = projected_polygon.simplify(1e-5, preserve_topology=True)

    # Extract coordinates from cleaned, transformed polygon.
    projected_coords = np.array(projected_polygon.exterior.coords)

    boundary_path = mpath.Path(projected_coords)

    return boundary_path

def get_axes(fig_kwargs={}, gridlines_kwargs={}, coastlines=True, gridlines=True, custom_boundary_extents=None, custom_boundary=True):

    fig_kwargs_ = dict(
        figsize=(10, 6),
        sharey=True,
        layout="constrained",
    )
    fig_kwargs_.update(fig_kwargs)

    geoaxes = False
    subplot_kw = fig_kwargs_.get("subplot_kw", None)
    if subplot_kw:
        projection = subplot_kw.get("projection", None)
        if projection:
            geoaxes = True

    gridlines_kwargs_ = dict(
        draw_labels=True,
        dms=True,
        auto_inline=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
        crs=ccrs.PlateCarree(),
        # x_inline=False,
        # y_inline=True,
    )
    gridlines_kwargs_.update(gridlines_kwargs)

    fig, axes = plt.subplots(
        **fig_kwargs_,
    )

    if custom_boundary:
        # Set extent to northern hemisphere if not provided.
        # This should be fixed for canari-ml
        if not custom_boundary_extents:
            lon_min, lon_max = -180, 180
            lat_min, lat_max = 0, 90
        region_path = high_res_rectangle(
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            target_crs=axes[0].projection,
            num_points=200,
        )

    for ax in axes.flat:
        if geoaxes:
            if coastlines:
                ax.coastlines(resolution="110m", linewidth=1, color="black")
            # ax.add_feature(land, linewidth=0.2, linestyle='--', edgecolor='k', alpha=1)
            if gridlines:
                ax.gridlines(**gridlines_kwargs)

            if custom_boundary:
                # Set the boundary on the axes without a further transform,
                # since the boundary is already in the axes' coordinate system.
                ax.set_boundary(region_path)

                # ax.set_frame_on(False)  # Hide boundary frame, else drawing line by longitude boundary

    return fig, axes
