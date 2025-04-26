import argparse
import logging

from canari_ml.utils import date_arg, location_arg, region_arg
from preprocess_toolbox.cli import BaseArgParser


class PlottingNumpyArgParser(BaseArgParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("numpy_file", type=str)

class PlottingArgParser(BaseArgParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("numpy_file", type=str)

class ForecastPlotArgParser(argparse.ArgumentParser):
    """An ArgumentParser specialised to support forecast plot arguments

    Additional argument enabled by allow_ecmwf() etc.

    The 'allow_*' methods return self to permit method chaining.

    :param forecast_date: allows this positional argument to be disabled
    """

    def __init__(self, *args, forecast_date: bool = True, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("obs_data_config", type=str, help="Path to the source data config file.")
        self.add_argument("forecast_file", type=str)
        if forecast_date:
            self.add_argument("forecast_date", type=date_arg)

        self.add_argument("-o", "--output-path", type=str, default=None)
        self.add_argument("-v",
                          "--verbose",
                          action="store_true",
                          default=False)
        self.add_argument("-r",
                          "--region",
                          default=None,
                          type=region_arg,
                          help="Region specified x1, y1, x2, y2")
        self.add_argument("--show-plot",
                          action="store_true",
                          default=False,
                          help="Show the plot instead of saving as video to file.")

    def allow_ecmwf(self):
        self.add_argument("-b",
                          "--bias-correct",
                          help="Bias correct SEAS forecast array",
                          action="store_true",
                          default=False)
        self.add_argument("-e", "--ecmwf", action="store_true", default=False)
        return self

    def allow_threshold(self):
        self.add_argument("-t",
                          "--threshold",
                          help="The SIC threshold of interest",
                          type=float,
                          default=0.15)
        return self

    def allow_sie(self):
        self.add_argument(
            "-ga",
            "--grid-area",
            help="The length of the sides of the grid used (in km)",
            type=int,
            default=25)
        return self

    def allow_metrics(self):
        self.add_argument("-m",
                          "--metrics",
                          help="Which metrics to compute and plot",
                          type=str,
                          default="mae,mse,rmse")
        self.add_argument(
            "-s",
            "--separate",
            help="Whether or not to produce separate plots for each metric",
            action="store_true",
            default=False)
        return self

    def allow_probes(self):
        self.add_argument(
            "-p",
            "--probe",
            action="append",
            dest="probes",
            type=location_arg,
            metavar="LOCATION",
            help="Sample at LOCATION",
        )
        return self

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)

        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            force=True,
        )
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        return args
