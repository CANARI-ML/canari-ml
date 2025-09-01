import argparse
import glob
import inspect
import logging
import os

import dask
import numpy as np
import orjson
import pandas as pd
import tensorflow as tf
from download_toolbox.interface import DataCollection
from icenet.data.datasets.utils import get_decoder
from icenet.data.loader import IceNetDataLoaderFactory
from icenet.data.loaders.base import IceNetBaseDataLoader
from torch.utils.data import Dataset
import icenet

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

        # Add file paths to the train, validation, and test datasets
        >>> split_dataset.add_records(base_path="./network_datasets/notebook_data/")
    """
    _batch_size: int
    _dtype: object
    _num_channels: int
    _lead_time: int
    _shape: int
    _shuffling: bool

    train_fns = []
    test_fns = []
    val_fns = []

    def add_records(self, base_path: str) -> None:
        """Add list of paths to train, val, test *.tfrecord(s) to relevant instance attributes.

        Add sorted list of file paths to train, validation, and test datasets in SplittingMixin.

        Args:
            base_path (str): The base path where the datasets are located.

        Returns:
            None. Updates `self.train_fns`, `self.val_fns`, `self.test_fns` with list
                of *.tfrecord files.
        """
        train_path = os.path.join(base_path, "train")
        val_path = os.path.join(base_path, "val")
        test_path = os.path.join(base_path, "test")

        logging.info("Training dataset path: {}".format(train_path))
        self.train_fns += sorted(glob.glob("{}/*.tfrecord".format(train_path)))
        logging.info("Validation dataset path: {}".format(val_path))
        self.val_fns += sorted(glob.glob("{}/*.tfrecord".format(val_path)))
        logging.info("Test dataset path: {}".format(test_path))
        self.test_fns += sorted(glob.glob("{}/*.tfrecord".format(test_path)))

    def get_split_datasets(self, ratio: object = None):
        """Retrieves train, val, and test datasets from corresponding attributes of SplittingMixin.

        Retrieves the train, validation, and test datasets from the file paths stored in the
            `train_fns`, `val_fns`, and `test_fns` attributes of SplittingMixin.

        Args:
            ratio (optional): A float representing the truncated list of datasets to be used.
                If not specified, all datasets will be used.
                Defaults to None.

        Returns:
            tuple: A tuple containing the train, validation, and test datasets.

        Raises:
            RuntimeError: If no files have been found in the train, validation, and test datasets.
            RuntimeError: If the ratio is greater than 1.
        """
        if not (len(self.train_fns) + len(self.val_fns) + len(self.test_fns)):
            raise RuntimeError("No files have been found, abandoning. This is "
                               "likely because you're trying to use a config "
                               "only mode dataset in a situation that demands "
                               "tfrecords to be generated (like training...)")

        logging.info("Datasets: {} train, {} val and {} test filenames".format(
            len(self.train_fns), len(self.val_fns), len(self.test_fns)))

        # If ratio is specified, truncate file paths for train, val, test using the ratio.
        if ratio:
            if ratio > 1.0:
                raise RuntimeError("Ratio cannot be more than 1")

            logging.info("Reducing datasets to {} of total files".format(ratio))
            train_idx, val_idx, test_idx = \
                int(len(self.train_fns) * ratio), \
                int(len(self.val_fns) * ratio), \
                int(len(self.test_fns) * ratio)

            if train_idx > 0:
                self.train_fns = self.train_fns[:train_idx]
            if val_idx > 0:
                self.val_fns = self.val_fns[:val_idx]
            if test_idx > 0:
                self.test_fns = self.test_fns[:test_idx]

            logging.info(
                "Reduced: {} train, {} val and {} test filenames".format(
                    len(self.train_fns), len(self.val_fns), len(self.test_fns)))

        # Loads from files as bytes exactly as written. Must parse and decode it.
        train_ds, val_ds, test_ds = \
            tf.data.TFRecordDataset(self.train_fns,
                                    num_parallel_reads=self.batch_size), \
            tf.data.TFRecordDataset(self.val_fns,
                                    num_parallel_reads=self.batch_size), \
            tf.data.TFRecordDataset(self.test_fns,
                                    num_parallel_reads=self.batch_size),

        # TODO: Comparison/profiling runs
        # TODO: parallel for batch size while that's small
        # TODO: obj.decode_item might not work here - figure out runtime
        #  implementation based on wrapped function call that can be serialised
        decoder = get_decoder(self.shape,
                              self.num_channels,
                              self.lead_time,
                              dtype=self.dtype.__name__)

        if self.shuffling:
            logging.info("Training dataset(s) marked to be shuffled")
            # FIXME: this is not a good calculation, but we don't have access
            #  in the mixin to the configuration that generated the dataset #57
            train_ds = train_ds.shuffle(
                min(int(len(self.train_fns) * self.batch_size), 366))

        # Since TFRecordDataset does not parse or decode the dataset from bytes,
        # use custom decoder function with map to do so.
        train_ds = train_ds.\
            map(decoder, num_parallel_calls=self.batch_size).\
            batch(self.batch_size)

        val_ds = val_ds.\
            map(decoder, num_parallel_calls=self.batch_size).\
            batch(self.batch_size)

        test_ds = test_ds.\
            map(decoder, num_parallel_calls=self.batch_size).\
            batch(self.batch_size)

        return train_ds.prefetch(tf.data.AUTOTUNE), \
            val_ds.prefetch(tf.data.AUTOTUNE), \
            test_ds.prefetch(tf.data.AUTOTUNE)

    def check_dataset(self, split: str = "train") -> None:
        """Check the dataset for NaN, log debugging info regarding dataset shape and bounds.

        Also logs a warning if any NaN are found.

        Args:
            split: The split of the dataset to check. Default is "train".
        """
        logging.debug("Checking dataset {}".format(split))

        decoder = get_decoder(self.shape,
                              self.num_channels,
                              self.lead_time,
                              dtype=self.dtype.__name__)

        for df in getattr(self, "{}_fns".format(split)):
            logging.info("Getting records from {}".format(df))
            try:
                raw_dataset = tf.data.TFRecordDataset([df])
                raw_dataset = raw_dataset.map(decoder)

                for i, (x, y, sw) in enumerate(raw_dataset):
                    x = x.numpy()
                    y = y.numpy()
                    sw = sw.numpy()

                    logging.debug(
                        "Got record {}:{} with x {} y {} sw {}".format(
                            df, i, x.shape, y.shape, sw.shape))

                    input_nans = np.isnan(x).sum()
                    output_nans = np.isnan(y[(sw > 0.)]).sum()
                    sw_nans = np.isnan(sw).sum()
                    input_min = np.min(x)
                    input_max = np.max(x)
                    output_min = np.min(x)
                    output_max = np.max(x)
                    sw_min = np.min(x)
                    sw_max = np.max(x)

                    logging.debug(
                        "Bounds: Input {}:{} Output {}:{} SW {}:{}".format(
                            input_min, input_max, output_min, output_max,
                            sw_min, sw_max))

                    if input_nans > 0:
                        logging.warning("Input NaNs detected in {}:{}".format(df, i))

                    if output_nans > 0:
                        logging.warning(
                            "Output NaNs detected in {}:{}, not accounted for by sample weighting".format(df, i))

                    if sw_nans > 0:
                        logging.warning(
                            "SW NaNs detected in {}:{}".format(df, i))
            except tf.errors.DataLossError as e:
                logging.warning("{}: data loss error {}".format(df, e.message))
            except tf.errors.OpError as e:
                logging.warning("{}: tensorflow error {}".format(df, e.message))
            # We don't except any non-tensorflow errors to prevent progression

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
        _generate_workers: An integer representing number of workers for parallel processing with Dask.
        _lead_time: An integer representing number of days to predict for.
        _num_channels: An integer representing number of channels (input variables) in the dataset.
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

        # Check JSON config has attribute for path to tfrecord datasets, and
        #   that the path exists.
        if self._config[path_attr] and \
                os.path.exists(self._config[path_attr]):
            self.add_records(self.path)
        else:
            logging.warning("Running in configuration only mode, tfrecords "
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
