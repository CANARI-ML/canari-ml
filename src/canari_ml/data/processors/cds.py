import logging

from preprocess_toolbox.processor import NormalisingChannelProcessor

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
