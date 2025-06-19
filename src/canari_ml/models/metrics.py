import torch
from torchmetrics.metric import Metric


class BaseMetric(Metric):
    """
    Base class for all metrics.

    Reference: https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
    """

    def __init__(self, leadtimes_to_evaluate: list = None):
        """Weighted metric for multiple leadtimes.

        Args:
            leadtimes_to_evaluate: A list of leadtimes to consider
                e.g., [0, 1, 2, 3, 4, 5] to consider first n days/months (i.e. leadtime)
                    in accuracy computation
                e.g., [0] to only look at the first day's accuracy
                e.g., [5] to only look at the sixth day's accuracy
        """
        super().__init__()
        self.leadtimes_to_evaluate = (
            leadtimes_to_evaluate if leadtimes_to_evaluate is not None else slice(None)
        )

    def _select_leadtimes(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> None:
        leadtime_index = self.leadtimes_to_evaluate
        predictions = predictions[..., leadtime_index]
        targets = targets[..., leadtime_index]
        sample_weight = sample_weight[..., leadtime_index]
        return predictions, targets, sample_weight


class MAE(BaseMetric):
    """
    Weighted MAE metric for use at multiple leadtimes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state(
            "sum_weighted_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total_weight", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor,
        **kwargs,
    ) -> None:
        """Update state with predictions and targets."""
        predictions, targets, sample_weight = self._select_leadtimes(
            predictions, targets, sample_weight
        )

        abs_error = (predictions - targets).abs()
        weighted_sum_abs_error = (abs_error * sample_weight).sum()
        total_weight = sample_weight.sum()

        self.sum_weighted_abs_error += weighted_sum_abs_error
        self.total_weight += total_weight

    def compute(self) -> torch.Tensor:
        return self.sum_weighted_abs_error / self.total_weight


class MSE(BaseMetric):
    """
    Weighted MSE metric for use at multiple leadtimes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state(
            "sum_weighted_squared_error",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state("total_weight", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor,
        **kwargs,
    ) -> None:
        """Update state with predictions and targets."""
        predictions, targets, sample_weight = self._select_leadtimes(
            predictions, targets, sample_weight
        )

        squared_error = (predictions - targets) ** 2.0
        weighted_sum_squared_error = (squared_error * sample_weight).sum()
        total_weight = sample_weight.sum()

        self.sum_weighted_squared_error += weighted_sum_squared_error
        self.total_weight += total_weight

    def compute(self) -> torch.Tensor:
        return self.sum_weighted_squared_error / self.total_weight


class RMSE(MSE):
    """
    Weighted Root Mean Squared Error for use at multiple leadtimes., computed as sqrt
    of Weighted MSE.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self) -> torch.Tensor:
        return super().compute().sqrt()
