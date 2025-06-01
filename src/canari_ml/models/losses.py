import torch.nn as nn

LOSS_REGISTRY = {
    "l1": nn.L1Loss,
    "mse": nn.MSELoss,
    "huber": nn.HuberLoss,
}


class WeightedLoss(nn.Module):
    """
    Weighted loss.

    Compute loss weighted by masking.
    """

    reduction: str

    def __init__(self, loss_type="mse", **kwargs) -> None:
        super().__init__()

        if loss_type not in LOSS_REGISTRY:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        self.loss_fn = LOSS_REGISTRY[loss_type.lower()](reduction="none", **kwargs)

    def forward(self, predictions, targets, sample_weights):
        loss = self.loss_fn(predictions, targets) * sample_weights

        return loss.mean()
