import torch.nn as nn


class L1Loss(nn.L1Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, targets, sample_weights):
        """
        Weighted L1 loss.

        Compute L1 loss weighted by masking.

        """
        y_hat = inputs

        # Computing using nn.L1Loss class. This class must be instantiated via:
        # >>> criterion = WeightedL1Loss(reduction="none")
        loss = super().forward(
            (100 * y_hat.movedim(-2, 1)), (100 * targets.movedim(-1, 1))
        ) * sample_weights.movedim(-1, 1)

        # Computing here, in the derived class
        # loss = (
        #             torch.abs( ( y_hat.movedim(-2, 1) - targets.movedim(-1, 1) )*100 )
        #         )*sample_weights.movedim(-1, 1)

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
        y_hat = inputs

        # Computing using nn.MSELoss base class. This class must be instantiated via:
        # criterion = nn.MSELoss(reduction="none")
        loss = super().forward(
            (100 * y_hat.movedim(-2, 1)), (100 * targets.movedim(-1, 1))
        ) * sample_weights.movedim(-1, 1)

        # Computing here, in the nn.Module derived class
        # loss = (
        #             ( ( y_hat.movedim(-2, 1) - targets.movedim(-1, 1) )*100 )**2
        #         )*sample_weights.movedim(-1, 1)
        return loss.mean()
