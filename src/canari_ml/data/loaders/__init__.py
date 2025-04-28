import inspect

from icenet.data.loaders.base import IceNetBaseDataLoader
from canari_ml.data.loaders.serial import SerialLoader

class CanariMLDataLoaderFactory:
    """A factory class for managing a map of loader names and their corresponding 
        implementation classes.
        Based on [IceNet v0.4.0_dev](https://github.com/icenet-ai/icenet/blob/6caa234907904bfa76b8724d8c83cd989230494a/icenet/data/loaders/__init__.py)

    Attributes:
        _loader_map: A dictionary holding loader names against their implementation
            classes.
    """

    def __init__(self):
        """Initialises the CanariMLDataLoaderFactory instance and sets up the initial
            loader map.
        """
        self._loader_map = dict(
            serial=SerialLoader,
            # dask=DaskMultiWorkerLoader, # TODO: Implement this
        )

    def add_data_loader(self, loader_name: str, loader_impl: object) -> None:
        """Adds a new loader to the loader map with the given name and implementation
            class.

        Args:
            loader_name: The name of the loader.
            loader_impl: The implementation class of the loader.

        Returns:
            None. Updates `_loader_map` attribute in CanariMLDataLoaderFactory with
                specified loader name and implementation.

        Raises:
            RuntimeError: If the loader name already exists or if the implementation
                class is not a descendant of IceNetBaseDataLoader.
        """
        if loader_name not in self._loader_map:
            if IceNetBaseDataLoader in inspect.getmro(loader_impl):
                self._loader_map[loader_name] = loader_impl
            else:
                raise RuntimeError(
                    "{} is not descended from IceNetBaseDataLoader".format(
                        loader_impl.__name__
                    )
                )
        else:
            raise RuntimeError(
                "Cannot add {} as already in loader map".format(loader_name)
            )

    def create_data_loader(self, loader_name, *args, **kwargs) -> object:
        """Creates an instance of a loader based on specified name from the
           `_loader_map` dict attribute.

        Args:
            loader_name: The name of the loader.
            *args: Additional positional arguments, is passed to the loader constructor.
            **kwargs: Additional keyword arguments, is passed to the loader constructor.

        Returns:
            An instance of the loader class.

        Raises:
            KeyError: If the loader name does not exist in `_loader_map`.
        """
        return self._loader_map[loader_name](*args, **kwargs)

    @property
    def loader_map(self) -> dict:
        """The loader map dictionary."""
        return self._loader_map
