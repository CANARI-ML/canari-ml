import torch.nn as nn


class L1Loss(nn.L1Loss):
    def __init__(self, *args, **kwargs):
        if "reduction" not in kwargs:
            kwargs.update({"reduction": "none"})
        super().__init__(*args, **kwargs)

    def forward(self, inputs, targets, sample_weights):
        """
        Weighted L1 loss.

        Compute L1 loss weighted by masking.

        """
        loss = super().forward(inputs, targets) * sample_weights

        return loss.mean()


class MSELoss(nn.MSELoss):
    def __init__(self, *args, **kwargs):
        if "reduction" not in kwargs:
            kwargs.update({"reduction": "none"})
        super().__init__(*args, **kwargs)

    def forward(self, inputs, targets, sample_weights):
        """
        Weighted MSE loss.

        Compute MSE loss weighted by masking.

        """
        loss = super().forward(inputs, targets)
        loss *= sample_weights
        return loss.mean()


class HuberLoss(nn.HuberLoss):
    def __init__(self, *args, **kwargs):
        if "reduction" not in kwargs:
            kwargs.update({"reduction": "none"})
        super().__init__(*args, **kwargs)

    def forward(self, inputs, targets, sample_weights):
        """
        Weighted Huber loss.

        Compute Huber loss weighted by masking.

        """
        loss = super().forward(inputs, targets)
        loss *= sample_weights
        return loss.mean()
