import logging
import os

import numpy as np
import tensorflow as tf
import torch
import zarr
from icenet.data.network_dataset import IceNetDataSet
from icenet.data.datasets.utils import get_decoder
from download_toolbox.interface import DataCollection
from torch.utils.data import DataLoader, Dataset
from canari_ml.data.loaders import CanariMLDataLoaderFactory

# TODO: Decide which dataloader class will be used in this project.

class TFRecordDataset(Dataset):
    def __init__(self, file_paths, decoder):
        self.file_paths = file_paths
        self.decoder = decoder
        self.raw_dataset = tf.data.TFRecordDataset(self.file_paths)
        self.length = sum(1 for _ in self.raw_dataset)

        # self.dataset = self.raw_dataset.map(self.decoder)
        # self.data = list(self.dataset.as_numpy_iterator())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        raw_dataset = self.raw_dataset.skip(idx).take(1)

        # print(type(raw_dataset))
        # x, y, sample_weights = list(raw_dataset.as_numpy_iterator())[0]
        x, y, sample_weights = list(raw_dataset.map(self.decoder))[0]

        # return x, y, sample_weights
        return x.numpy(), y.numpy(), sample_weights.numpy()

        # x, y, sample_weights = self.data[idx]
        # return x, y, sample_weights
        # return torch.tensor(x), torch.tensor(y), torch.tensor(sample_weights)

        # x, y, sample_weights = torch.tensor(sample["x"], dtype=torch.float32)
        # # raw_dataset = tf.data.TFRecordDataset(self.file_paths[idx])
        # # dataset = raw_dataset.map(self.decoder)
        # # x, y, sample_weights = list(self.dataset)
        # dataset = self.dataset.skip(idx).take(1)
        # x, y, sample_weights = self.dataset
        # return x.numpy(), y.numpy(), sample_weights.numpy()

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)


class ZarrDataset(Dataset):
    def __init__(
        self, root_path: str, zarr_name: str, train_split: bool = True
    ) -> None:
        """
        Initialize the dataset from a directory containing Zarr files.

        Args:
            root_path: Path to the directory containing 'train.zarr', 'val.zarr', 'test.zarr'.
            zarr_name: Name of the Zarr file to load (e.g., 'train.zarr', 'val.zarr', 'test.zarr').
            train_split: Whether to load the training split
                Defaults to True.
        """
        self.root_path = root_path
        self.train_split = train_split

        zarr_path = os.path.join(root_path, zarr_name)

        self.store = zarr.open(zarr_path)
        self.x_array = self.store["x"]
        self.y_array = self.store["y"]
        self.sw_array = self.store.get("sample_weights", None)

    def __len__(self) -> int:
        """
        Return the number of samples (dates) in the dataset.
        """
        return len(self.x_array)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a single sample from the Zarr store.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing 'x' tensor, 'y' tensor, and optional
            'sample_weights' tensor.
        """
        x = self.x_array[idx]
        y = self.y_array[idx]

        # Convert to PyTorch tensors if needed. Assuming numpy arrays are already correct dtype.
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).float()

        sample_weights = None
        if self.sw_array is not None:
            sw = self.sw_array[idx]
            sample_weights = torch.from_numpy(sw).float()

        return {"x": x_tensor, "y": y_tensor, "sample_weights": sample_weights}


class CANARIMLDataSetTorch(IceNetDataSet):
    def __init__(
        self,
        configuration_path,
        *args,
        batch_size=4,
        path=os.path.join(".", "network_datasets"),
        shuffling=False,
        **kwargs,
    ):
        super().__init__(configuration_path=configuration_path)

        self._config = {}
        self._configuration_path = configuration_path
        self._load_configuration(configuration_path)

        self._batch_size = batch_size
        self._lead_time = self._config["lead_time"]
        self._num_channels = self._config["num_channels"]
        self._shape = tuple(self._config["shape"])
        self._shuffling = shuffling
        self.hemi = "south" if self._config["south"] else "north" if self._config["north"] else None

        if self._config.get("dataset_path") and os.path.exists(
            self._config["dataset_path"]
        ):
            if not self.train_fns or not self.val_fns or not self.test_fns:
                self.add_records(self.base_path)
        else:
            logging.warning(
                "Running in configuration only mode, tfrecords were not generated for this dataset"
            )

    def get_data_loaders(self, ratio=None):

        num_workers = 4
        persistent_workers = True if num_workers else False

        root_path = self._config["dataset_path"]

        train_dataset = ZarrDataset(root_path=os.path.join(root_path, "train"), zarr_name="train.zarr")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffling,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True,  # For faster transfer to GPU if using one
        )

        val_dataset = ZarrDataset(root_path=os.path.join(root_path, "val"), zarr_name="val.zarr")
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True,
        )

        test_dataset = ZarrDataset(root_path=os.path.join(root_path, "test"), zarr_name="test.zarr")
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True,
        )
        return train_dataloader, val_dataloader, test_dataloader

    def get_data_loader(self,
                        lead_time: object = None,
                        generate_workers: object = None) -> object:
        """Create an instance of the CANARIDataLoader class.

        Args:
            lead_time (optional): The number of forecast steps to be used by the data loader.
                If not provided, defaults to the value specified in the configuration file.
            generate_workers (optional): An integer representing number of workers to use for
                parallel processing with Dask. If not provided, defaults to the value specified in
                the configuration file.

        Returns:
            An instance of the SerialLoader class configured with the specified parameters.
        """
        if lead_time is None:
            lead_time = self._config["lead_time"]
        if generate_workers is None:
            generate_workers = self._config["generate_workers"]
        loader = CanariMLDataLoaderFactory().create_data_loader(
            "serial",  # This will load the `SerialLoader` class.
            self.loader_config,
            self.identifier,
            lag_time=self._config["lag_time"],
            lead_time=lead_time,
            generate_workers=generate_workers,
            dataset_config_path=os.path.dirname(self._configuration_path),
            loss_weight_days=self._config["loss_weight_days"],
            output_batch_size=self._config["output_batch_size"],
            var_lag_override=self._config["var_lag_override"],
        )
        return loader

class IceNetDataSetTorch(IceNetDataSet):
    def __init__(
        self,
        configuration_path,
        *args,
        batch_size=4,
        path=os.path.join(".", "network_datasets"),
        shuffling=False,
        **kwargs,
    ):
        super().__init__(configuration_path=configuration_path)

        self._config = {}
        self._configuration_path = configuration_path
        self._load_configuration(configuration_path)

        # super(IceNetDataSetTorch, self).__init__(*args, identifier=self._config["identifier"], north=bool(self._config["north"]), path=path, south=bool(self._config["south"]), **kwargs)
        DataCollection.__init__(
            self,
            *args,
            identifier=self._config["identifier"],
            north=bool(self._config["north"]),
            path=path,
            south=bool(self._config["south"]),
            **kwargs,
        )

        self._batch_size = batch_size
        self._dtype = getattr(np, self._config["dtype"])
        self._n_forecast_days = self._config["n_forecast_days"]
        self._num_channels = self._config["num_channels"]
        self._shape = tuple(self._config["shape"])
        self._shuffling = shuffling

        if self._config.get("dataset_path") and os.path.exists(
            self._config["dataset_path"]
        ):
            hemi = self.hemisphere_str[0]
            if not self.train_fns or not self.val_fns or not self.test_fns:
                self.add_records(self.base_path, hemi)
        else:
            logging.warning(
                "Running in configuration only mode, tfrecords were not generated for this dataset"
            )

    def get_data_loaders(self, ratio=None):
        # train_ds, val_ds, test_ds = self.get_split_datasets(ratio)

        # Wrap TensorFlow datasets with TFRecordDataset
        decoder = get_decoder(
            self._shape,
            self._num_channels,
            self._n_forecast_days,
            dtype=self._dtype.__name__,
        )

        train_dataset = TFRecordDataset(self.train_fns, decoder)
        val_dataset = TFRecordDataset(self.val_fns, decoder)
        test_dataset = TFRecordDataset(self.test_fns, decoder)

        num_workers = 0
        persistent_workers = True if num_workers else False
        timeout = 30

        # Create PyTorch DataLoader instances
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffling,
            num_workers=num_workers,
            #   multiprocessing_context="spawn",
            persistent_workers=persistent_workers,
            #   timeout=timeout,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=num_workers,
            # multiprocessing_context="spawn",
            persistent_workers=persistent_workers,
            # timeout=timeout,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=num_workers,
            #  multiprocessing_context="spawn",
            persistent_workers=persistent_workers,
            #  timeout=timeout,
        )

        return train_loader, val_loader, test_loader
