import logging
import os

import numpy as np
import orjson
from download_toolbox.interface import DataCollection

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


# TODO: define a decent interface and sort the inheritance architecture out, as
#  this will facilitate the new datasets in #35
class SplittingMixin:
    """Read train, val, test datasets from tfrecord protocol buffer files.

    Split and shuffle data if specified as well.

    Example:
        This mixin is not to be used directly, but to give an idea of its use:

        # Initialise SplittingMixin
        >>> split_dataset = SplittingMixin()
    """
    _batch_size: int
    _dtype: object
    _num_channels: int
    _lead_time: int
    _shape: int
    _shuffling: bool

    @property
    def batch_size(self) -> int:
        """The dataset's batch size."""
        return self._batch_size

    @property
    def dtype(self) -> str:
        """The dataset's data type."""
        return self._dtype

    @property
    def lead_time(self) -> int:
        """The number of time steps to forecast."""
        return self._lead_time

    @property
    def num_channels(self) -> int:
        """The number of channels in dataset."""
        return self._num_channels

    @property
    def shape(self) -> object:
        """The shape of dataset."""
        return self._shape

    @property
    def shuffling(self) -> bool:
        """A flag for whether training dataset(s) are marked to be shuffled."""
        return self._shuffling


class IceNetDataSet(SplittingMixin, DataCollection):
    """Initialises and configures a dataset.

    It loads a JSON configuration file, updates the `_config` attribute with the
    result, creates a data loader, and methods to access the dataset.

    Attributes:
        _config: A dict used to store configuration loaded from JSON file.
        _configuration_path: The path to the JSON configuration file.
        _batch_size: The batch size for the data loader.
        _counts: A dict with number of elements in train, val, test.
        _dtype: The type of the dataset.
        _loader_config: The path to the data loader configuration file.
        _generate_workers: An integer representing number of workers for parallel
            processing with Dask.
        _lead_time: An integer representing number of days to predict for.
        _num_channels: An integer representing number of channels (input variables)
            in the dataset.
        _shape: The shape of the dataset.
        _shuffling: A flag indicating whether to shuffle the data or not.
    """

    def __init__(self,
                 configuration_path: str,
                 *args,
                 batch_size: int = 4,
                 path: str = os.path.join(".", "network_datasets"),
                 shuffling: bool = False,
                 **kwargs) -> None:
        """Initialises an instance of the IceNetDataSet class.

        Args:
            configuration_path: The path to the JSON configuration file.
            *args: Additional positional arguments.
            batch_size (optional): How many samples to load per batch. Defaults to 4.
            path (optional): The path to the directory where the processed tfrecord
                protocol buffer files will be stored. Defaults to './network_datasets'.
            shuffling (optional): Flag indicating whether to shuffle the data.
                Defaults to False.
            *args: Additional keyword arguments.
        """

        self._config = dict()
        self._configuration_path = configuration_path
        self._load_configuration(configuration_path)

        super().__init__(*args,
                         identifier=self._config["identifier"],
                         base_path=path,
                         **kwargs)

        # TODO: code smell - loading config twice because not using DataCollection
        self._config = dict()
        self._load_configuration(configuration_path)
        self._batch_size = batch_size
        self._counts = self._config["counts"]
        self._dtype = getattr(np, self._config["dtype"])
        self._loader_config = self._config["loader_config"]
        self._generate_workers = self._config["generate_workers"]
        self._lead_time = self._config["lead_time"]
        self._num_channels = self._config["num_channels"]
        self._shape = tuple(self._config["shape"])
        self._shuffling = shuffling

        path_attr = "dataset_path"

        # Check JSON config has attribute for path to zarr datasets, and
        #   that the path exists.
        if self._config[path_attr] and \
                os.path.exists(self._config[path_attr]):
            pass
        else:
            logging.warning("Running in configuration only mode, Zarr datasets"
                            "were not generated for this dataset")

    def _load_configuration(self, path: str) -> None:
        """Load the JSON configuration file and update the `_config` attribute of `IceNetDataSet` class.

        Args:
            path: The path to the JSON configuration file.

        Raises:
            OSError: If the specified configuration file is not found.
        """
        if os.path.exists(path):
            logging.info("Loading configuration {}".format(path))

            with open(path, "r") as fh:
                obj = orjson.loads(fh.read())

                self._config.update(obj)
        else:
            raise OSError("{} not found".format(path))

    @property
    def loader_config(self) -> str:
        """The path to the JSON loader configuration file stored in the dataset config file."""
        # E.g. `/path/to/loader.{identifier}.json`
        return self._loader_config

    @property
    def channels(self) -> list:
        """The list of channels (variable names) specified in the dataset config file."""
        return self._config["channels"]

    @property
    def counts(self) -> dict:
        """A dict with number of elements in train, val, test in the config file."""
        return self._config["counts"]
