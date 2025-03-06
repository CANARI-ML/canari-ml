from preprocess_toolbox.cli import BaseArgParser


class PlottingNumpyArgParser(BaseArgParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("numpy_file", type=str)

class PlottingArgParser(BaseArgParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("numpy_file", type=str)
