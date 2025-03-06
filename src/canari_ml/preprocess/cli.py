from preprocess_toolbox.dataset.cli import ProcessingArgParser


class ReprojectArgParser(ProcessingArgParser):
    def __init__(self):
        super().__init__()

    def add_coarsen(self):
        self.add_argument(
            "--coarsen",
            type=int,
            default=1,
            help="To coarsen output grid by this integer factor.",
        )
        return self

    def add_interpolate(self):
        self.add_argument(
            "--interpolate",
            action="store_true",
            help="Enable nearest neighbour interpolation to fill in missing areas.",
        )
        return self
