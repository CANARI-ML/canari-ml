import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cartopy.crs as ccrs
import iris
import iris.coord_systems
import iris.coords
import iris.cube
import iris.time
import iris.util
import numpy as np
import rioxarray
import xarray as xr
from cf_units import Unit
from download_toolbox.interface import DatasetConfig
from ncdata.iris_xarray import cubes_from_xarray, cubes_to_xarray

from .utils import parse_shape

# Get the logger for rasterio (or the relevant library)
logger = logging.getLogger("rasterio")

# Set the logging level to WARNING to suppress INFO-level messages
logger.setLevel(logging.WARNING)


def ease2_reference_grid_setup(
    shape: tuple[int, int], target_crs: str
) -> tuple[iris.cube.Cube, iris.cube.Cube]:
    """
    Sets up the EASE-Grid 2.0 CRS and creates a grid and binary mask.

    This function defines the EASE-Grid 2.0 parameters based on the provided
    shape and target CRS. It creates a grid with projection coordinates and
    a binary mask that distinguishes the Northern Hemisphere (NH) from the
    Southern Hemisphere (SH).

    Args:
        shape: The shape of the grid (rows, columns).
            Supported shapes are (720, 720) for 25 km resolution and
            (500, 500) for 36 km resolution.
        target_crs: The target coordinate reference system (CRS) in
            EPSG format (e.g., "EPSG:6931" for EASE-Grid 2.0 Northern Hemisphere).

    Returns:
        tuple:
            - grid: An empty Iris cube representing the grid
              with projection coordinates.
            - mask: A binary mask cube with values 1 for the
              Northern Hemisphere and NaN for the Southern Hemisphere.

    Raises:
        ValueError: If the provided shape does not match the expected EASE-Grid
            2.0 standard grid or if the target CRS is not supported.

    References:
        - https://nsidc.org/data/user-resources/help-center/guide-ease-grids#anchor-25-km-resolution-ease-grids
    """

    if target_crs == "EPSG:6931" or target_crs == "EPSG:6932":
        # Define grid parameters for EASE-Grid 2.0 standard grid
        # Reference: https://nsidc.org/data/user-resources/help-center/guide-ease-grids#anchor-25-km-resolution-ease-grids
        # `grid_spacing` is the grid resolution in meters taken from the table in above link
        if shape == (720, 720):
            grid_spacing = 25000
        elif shape == (500, 500):
            grid_spacing = 36000
        else:
            raise ValueError(
                f"shape doesn't match expected EASE-Grid 2.0 standard grid:\n\t(`{shape}`)"
            )
        # The x-axis coordinate of the outer edge of the upper-left pixel
        x0 = -9000000.0
        # The y-axis coordinate of the outer edge of the upper-left pixel
        y0 = 9000000.0
    else:
        raise ValueError(
            f"target_crs doesn't match expected EASE-Grid 2.0 standard grid:\n\t(`{target_crs}`)"
        )

    # define grid spacing and size of grid in X and Y directions
    # see: https://nsidc.org/data/user-resources/help-center/guide-ease-grids#anchor-25-km-resolution-ease-grids
    num_grid_points = shape[0]

    target_crs = target_crs.split(":")[1] if target_crs else 6931 # type: ignore

    # use cartopy to read the EASE grid parameters based on the EPSG code
    # for a Northern Hemisphere, Lambert Azimuthal grid (as specified at the NSIDC link above)
    # this avoids having to hard-code any of the parameters
    ease2_nh = ccrs.epsg(target_crs)
    ease2_nh_params = ease2_nh.to_dict()

    # create the equivalent Iris ellipsoid and CRS using the cartopy metadata
    ellipsoid = iris.coord_systems.GeogCS(
        semi_major_axis=ease2_nh.ellipsoid.semi_major_metre,
        semi_minor_axis=ease2_nh.ellipsoid.semi_minor_metre,
    )
    crs = iris.coord_systems.LambertAzimuthalEqualArea(
        latitude_of_projection_origin=ease2_nh_params["lat_0"],
        longitude_of_projection_origin=ease2_nh_params["lon_0"],
        false_easting=ease2_nh_params["x_0"],
        false_northing=ease2_nh_params["y_0"],
        ellipsoid=ellipsoid,
    )

    # work out the maximum value of X and Y coordinates for grid cell centres
    # for the specified grid with coordinates 0, 0 at the grid centre
    xylim = (num_grid_points / 2.0 - 0.5) * grid_spacing

    # create X and Y coordinate values for the grid
    # note the addition of the false easting and northing from the grid parameters,
    # though these are actually 0 for this grid anyway
    x = iris.coords.DimCoord(
        np.arange(-xylim, xylim + grid_spacing / 2, grid_spacing)
        + ease2_nh_params["x_0"],
        standard_name="projection_x_coordinate",
        units="m",
        coord_system=crs,
    )
    y = iris.coords.DimCoord(
        np.arange(xylim, -xylim - grid_spacing / 2, -grid_spacing)
        + ease2_nh_params["y_0"],
        standard_name="projection_y_coordinate",
        units="m",
        coord_system=crs,
    )

    # create an empty cube representing the grid
    grid = iris.cube.Cube(
        np.empty((y.shape[0], x.shape[0])), dim_coords_and_dims=[(y, 0), (x, 1)] # type: ignore
    )

    # create a binary mask on the EASE grid, with values 1 == NH, 0 == SH
    xy = np.meshgrid(grid.coord(axis="x").points, grid.coord(axis="y").points) # type: ignore
    lonlat = ccrs.Geodetic().transform_points(
        grid.coord_system().as_cartopy_crs(), xy[0], xy[1] # type: ignore
    )
    lon = lonlat[:, :, 0]
    lat = lonlat[:, :, 1]

    mask = grid.copy()
    mask.data.fill(1)
    mask.data[lat < 0] = np.nan
    mask.rename("binary_mask")

    return grid, mask


def reproject_dataset(
    input_: str | Path | xr.Dataset | xr.DataArray,
    grid: iris.cube.Cube,
    mask: iris.cube.Cube,
    target_crs: str = "EPSG:6931",
):
    """
    Reprojects a source dataset from `EPSG:4326` (lat/lon) to a target CRS using Iris.

    Based on code from Tony Phillips (BAS), sent via email on 04/03/25.This function
    takes an input dataset (xarray Dataset, DataArray, or file path),reprojects it to
    the specified target CRS, and applies a binary mask. The output is returned as an
    xarray Dataset or DataArray.

    Args:
        input_: The input dataset to reproject.
            Can be a file path, an xarray Dataset, or an xarray DataArray.
        grid: The target grid as an Iris cube.
        mask: A binary mask cube with values 1 for the Northern Hemisphere
            and NaN for the Southern Hemisphere.
        target_crs (optional): The target coordinate reference system (CRS)
            in EPSG format (e.g., "EPSG:6931"). Defaults to "EPSG:6931".

    Returns:
        The reprojected dataset. If the input was an xarray DataArray, the output
            will also be a DataArray. Otherwise, it will be a Dataset.

    Raises:
        ValueError: If the input type is not a string path, xarray Dataset, or
            array DataArray.

    References:
        - https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#heading-SpatialreferencesystemsandEarthmodel
    """
    iris.FUTURE.save_split_attrs = True
    return_dataarray = False

    if isinstance(input_, xr.Dataset):
        ds = input_
    elif isinstance(input_, xr.DataArray):
        ds = input_.to_dataset()
        return_dataarray = True
    elif isinstance(input_, str) or isinstance(input_, Path):
        ds = xr.open_dataset(input_, decode_coords="all", engine="h5netcdf")
    else:
        raise ValueError(
            f"input {input_} must be a string path, xr.Dataset, or xr.DataArray, not {type(input)}"
        )

    (cube,) = cubes_from_xarray(ds)

    # Extend the mask to include the time dimension
    time_dim = cube.coord_dims("time")[0]
    time_len = cube.shape[time_dim]
    mask_data = np.broadcast_to(mask.data, (time_len,) + mask.data.shape)

    # Ensure the cube coordinates have the EPSG:4326 CRS
    # Define EPSG:4326 CRS
    # Ref: https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#heading-SpatialreferencesystemsandEarthmodel
    # Assign EPSG:4326 CRS and units to the cube's coordinates
    for axis in ["latitude", "longitude"]:
        cube.coord(axis).coord_system = iris.coord_systems.GeogCS(6371229.0)
        cube.coord(axis).units = Unit("degrees")
        cube.coord(axis).guess_bounds()

    # cube_reproject = cube.regrid(grid, iris.analysis.Nearest())
    cube_reproject = cube.regrid(
        grid, iris.analysis.Linear(extrapolation_mode="linear") # pyright: ignore[reportAttributeAccessIssue]
    )

    cube_reproject.data *= mask_data

    ds_reprojected = cubes_to_xarray(cube_reproject)
    ds_reprojected = ds_reprojected.rename(
        {"projection_x_coordinate": "x", "projection_y_coordinate": "y"}
    )
    ds_reprojected.attrs["grid_mapping"] = "spatial_ref"

    # Set the CRS for future rioxarray usage
    ds_reprojected.rio.write_crs(target_crs, inplace=True)

    if return_dataarray:
        varname = list(ds_reprojected.data_vars)[0]
        return ds_reprojected[varname]
    else:
        return ds_reprojected


def reproject_file(
    datafile: str, grid: iris.cube.Cube, mask: iris.cube.Cube, target_crs: str
) -> None:
    """
    Reprojects a single NetCDF file.

    Args:
        datafile: Path to the NetCDF file to reproject.

    Raises:
        Exception: If an error occurs during reprojection.
    """
    try:
        source_datafile = Path(datafile)
        (datafile_path, datafile_name) = os.path.split(datafile)
        reproject_source_name = f"_reproject_{datafile_name}"
        reproject_datafile = Path(datafile_path) / reproject_source_name
        source_datafile.rename(reproject_datafile)

        logging.debug(f"Reprojecting {reproject_datafile}")

        ds_reprojected = reproject_dataset(reproject_datafile, grid, mask, target_crs)

        logging.debug(f"Saving reprojected data to {datafile}... ")
        ds_reprojected.to_netcdf(datafile)
    except Exception as e:
        print(f"Error reprojecting {datafile}: {e}")
        raise
    finally:
        # Ensure temp file is deleted
        if os.path.exists(reproject_datafile):
            os.remove(reproject_datafile)


def reproject_datasets_from_config(
    process_config: DatasetConfig, workers: int = 1, **kwargs
) -> None:
    """
    Reprojects multiple datasets from input config file.

    Args:
        process_config: Configuration object containing dataset file paths.
        workers (optional): Number of parallel workers to use. Defaults to 1.
        **kwargs: Additional arguments to pass to `reproject_file`.
    """
    logging.info("Reprojecting dataset")

    shape = kwargs.get("shape", (720, 720))
    target_crs = kwargs["target_crs"]

    if not isinstance(shape, tuple):
        shape: tuple[int, int] = parse_shape(shape)

    grid, mask = ease2_reference_grid_setup(shape, target_crs)

    datafiles = [
        _ for var_files in process_config.var_files.values() for _ in var_files
    ]

    logging.info(f"{len(datafiles)} files to reproject")
    if workers > 1:
        logging.info(f"Reprojecting using {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(reproject_file, datafile, grid, mask, target_crs)
                for datafile in datafiles
            ]

        _ = [future.result() for future in futures]
    else:
        logging.info("Reprojecting using one worker")
        for datafile in datafiles:
            reproject_file(datafile, grid, mask, target_crs)

    logging.info("Reprojection completed")
