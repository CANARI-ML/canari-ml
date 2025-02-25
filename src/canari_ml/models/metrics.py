import torch
import torchmetrics
from torchmetrics.functional.regression.mae import _mean_absolute_error_update
from torchmetrics.functional.regression.mse import _mean_squared_error_update


class MAE(torchmetrics.MeanAbsoluteError):
    """MAE metric for use at multiple leadtimes.

    Reference: https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
    """

    def __init__(self, leadtimes_to_evaluate: list = None):
        """Weighted MAE metric for multiple leadtimes.

        Args:
            leadtimes_to_evaluate: A list of leadtimes to consider
                e.g., [0, 1, 2, 3, 4, 5] to consider first n days/months (i.e. leadtime)
                    in accuracy computation
                e.g., [0] to only look at the first day's accuracy
                e.g., [5] to only look at the sixth day's accuracy
        """
        # Pass `squared=False` to get RMSE instead of MSE
        super().__init__()
        self.leadtimes_to_evaluate = (
            leadtimes_to_evaluate if leadtimes_to_evaluate is not None else slice(None)
        )

    def update(self, preds, target, sample_weight: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        predictions = (preds * sample_weight)[..., self.leadtimes_to_evaluate]
        targets = (target * sample_weight)[..., self.leadtimes_to_evaluate]
        sum_abs_error, num_obs = _mean_absolute_error_update(
            predictions, targets, num_outputs=self.num_outputs
        )

        self.sum_abs_error += sum_abs_error
        self.total += num_obs


class MSE(torchmetrics.MeanSquaredError):
    """MSE metric for use at multiple leadtimes.

    Reference: https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
    """

    def __init__(self, leadtimes_to_evaluate: list = None):
        """Weighted MSE metric for multiple leadtimes.

        Args:
            leadtimes_to_evaluate: A list of leadtimes to consider
                e.g., [0, 1, 2, 3, 4, 5] to consider first n days/months (i.e. leadtime)
                    in accuracy computation
                e.g., [0] to only look at the first day's accuracy
                e.g., [5] to only look at the sixth day's accuracy
        """
        # Pass `squared=True` to get MSE instead of RMSE
        super().__init__(squared=True)
        self.leadtimes_to_evaluate = (
            leadtimes_to_evaluate if leadtimes_to_evaluate is not None else slice(None)
        )

    def update(self, preds, target, sample_weight: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        predictions = (preds * sample_weight)[..., self.leadtimes_to_evaluate]
        targets = (target * sample_weight)[..., self.leadtimes_to_evaluate]
        sum_squared_error, num_obs = _mean_squared_error_update(
            predictions, targets, num_outputs=self.num_outputs
        )

        self.sum_squared_error += sum_squared_error
        self.total += num_obs


class RMSE(torchmetrics.MeanSquaredError):
    """RMSE metric for use at multiple leadtimes.

    Reference: https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
    """

    def __init__(self, leadtimes_to_evaluate: list = None):
        """Weighted RMSE metric for multiple leadtimes.

        Args:
            leadtimes_to_evaluate: A list of leadtimes to consider
                e.g., [0, 1, 2, 3, 4, 5] to consider first n days/months (i.e. leadtime)
                    in accuracy computation
                e.g., [0] to only look at the first day's accuracy
                e.g., [5] to only look at the sixth day's accuracy
        """
        # Pass `squared=False` to get RMSE instead of MSE
        super().__init__(squared=False)
        self.leadtimes_to_evaluate = (
            leadtimes_to_evaluate if leadtimes_to_evaluate is not None else slice(None)
        )

    def update(self, preds, target, sample_weight: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        predictions = (preds * sample_weight)[..., self.leadtimes_to_evaluate]
        targets = (target * sample_weight)[..., self.leadtimes_to_evaluate]
        sum_squared_error, num_obs = _mean_squared_error_update(
            predictions, targets, num_outputs=self.num_outputs
        )

        self.sum_squared_error += sum_squared_error
        self.total += num_obs
