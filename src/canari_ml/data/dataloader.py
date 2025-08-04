import logging
import os

import torch
import zarr
from icenet.data.network_dataset import IceNetDataSet
from torch.utils.data import DataLoader, Dataset

from canari_ml.data.loaders import CanariMLDataLoaderFactory


class ZarrDataset(Dataset):
    def __init__(
        self, root_path: str, zarr_name: str, train_split: bool = True
    ) -> None:
        """
        Initialise the dataset from a directory containing Zarr files.

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
        super().__init__(configuration_path=configuration_path, path=path)

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
                "Running in configuration only mode, Zarr cache files are not being generated for this dataset"
            )

    def get_data_loaders(self, num_workers=4, ratio=None):
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
                        generate_workers: object = None,
                        base_path: str = os.path.join(".", "network_datasets"),
                        dummy: bool = False,
                        ) -> object:
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
            base_path=base_path,
            dummy=dummy,
        )
        return loader
