import logging
import os

import xarray as xr
from preprocess_toolbox.processor import NormalisingChannelProcessor

from canari_ml.preprocess.utils import get_nc_encoding

logger = logging.getLogger(__name__)


class ERA5PreProcessor(NormalisingChannelProcessor):
    """Based on `NormalisingChannelProcessor` class from preprocess-toolbox"""

    def __init__(self, *args, smooth_sigma=0, **kwargs):
        self.smooth_sigma = smooth_sigma
        super().__init__(*args, **kwargs)

    def pre_normalisation(self, var_name: str, da: object):
        if "expver" in da.coords:
            logger.warning("expvers {} in coordinates, will process out but "
                            "this needs further work: expver needs storing for "
                            "later overwriting".format(da.expver))

        # Checking if variable is geopotential, like: 'zg250', 'zg500'
        # i.e., zg + an integer number
        is_geopotential_var = False
        if var_name.startswith("zg"):
            is_geopotential_var = "".join(var_name.split("zg")).isdigit()

        if var_name == "tos":
            logger.debug("ERA5 regrid postprocessing replacing zeroes: {}".format(var_name))
            da = da.fillna(0)
        elif is_geopotential_var:
            # Convert from geopotential to geopotential height
            logger.debug("ERA5 additional regrid: {}".format(var_name))
            da /= 9.80665

        return da

    def post_normalisation(self, var_name: str, da: object):
        ## TODO: Redundant, refactor this in future
        logger.info("Renaming ERA5 spatial coordinates to match IceNet")
        if "x" in da.coords and "y" in da.coords:
            da = da.rename(dict(x="xc", y="yc"))
        return da

    def save_processed_file(self,
                            var_name: str,
                            name: str,
                            data: xr.Dataset | xr.DataArray,
                            convert: bool = True,
                            overwrite: bool = False) -> str:
        """Save processed data to netCDF file.

        Args:
            var_name: The name of the variable.
            name: The name of the file.
            data: The data to be saved.
            convert: Whether to convert data to the processors data type
            overwrite: Whether to overwrite extant files

        Returns:
            object: The path of the saved netCDF file.

        """
        file_path = os.path.join(self.path, name)
        if overwrite or not os.path.exists(file_path):
            logging.debug("Writing to {}".format(file_path))
            if convert:
                data = data.astype(self._dtype)
            encoding = get_nc_encoding(data)
            data.to_netcdf(file_path, engine="h5netcdf", encoding=encoding)

        if var_name not in self.processed_files.keys():
            self.processed_files[var_name] = list()

        if file_path not in self.processed_files[var_name]:
            logging.debug("Adding {} file: {}".format(var_name, file_path))
            self.processed_files[var_name].append(file_path)
        # else:
        #     logging.warning("{} already exists in {} processed list".format(file_path, var_name))
        return file_path
