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
        loss = super().forward(y_hat.movedim(-1, 1), targets.movedim(-1, 1)) * sample_weights.movedim(-1, 1)

        # Computing here, in the derived class
        # loss = (
        #             torch.abs( ( y_hat.movedim(-1, 1) - targets.movedim(-1, 1) )*100 )
        #         )*sample_weights.movedim(-1, 1)

        return loss.mean()


class MSELoss(nn.MSELoss):
    def __init__(self, *args, **kwargs):
        if "reduction" not in kwargs:
            kwargs.update({"reduction": "none"})
        super().__init__(*args, **kwargs)
        self.count = 1

    def forward(self, inputs, targets, sample_weights):
        """
        Weighted MSE loss.

        Compute MSE loss weighted by masking.

        """
        inputs, targets, sample_weights = inputs, targets, sample_weights
        print("inputs.shape:", inputs.shape)
        print("targets.shape:", targets.shape)
        print("sample_weights.shape:", sample_weights.shape)
        y_hat = inputs.movedim(-2, 1).movedim(-1, 1)
        targets = targets.movedim(-1, 1).movedim(-1, 1)
        sample_weights = sample_weights.movedim(-1, 1).movedim(-1, 1)
        print("inputs.shape:", y_hat.shape)
        print("targets.shape:", targets.shape)
        print("sample_weights.shape:", sample_weights.shape)
        #exit()

        y_hat = y_hat * sample_weights
        targets = targets * sample_weights

        #if self.count % 100 == 0:
        #    import matplotlib.pyplot as plt
        #    #img1 = y_hat[0, :, :, 0, 0].cpu().data.numpy()
        #    #img2 = targets[0, :, :, 0, 0].cpu().data.numpy()
        #    img1 = y_hat[0, 0, 0, :, :].cpu().data.numpy()
        #    img2 = targets[0, 0, 0, :, :].cpu().data.numpy()
        #    print("img1 shape:", img1.shape)
        #    print("img2 shape:", img2.shape)
        #    fig, axes = plt.subplots(2, 2, figsize=(12, 6), layout="constrained")
        #    # Prediction at leadtime 0
        #    ## Plot current prediction
        #    im1 = axes[0, 0].imshow(img1, cmap="RdBu_r")
        #    plt.colorbar(im1, ax=axes[0, 0])
        #    axes[0, 0].set_title("y_hat: leadtime 0")
        #    ## Plot target
        #    im2 = axes[1, 0].imshow(img2, cmap="RdBu_r")
        #    plt.colorbar(im2, ax=axes[1, 0])
        #    axes[1, 0].set_title("target: leadtime 0")
        #    # Prediction at last leadtime
        #    img1 = y_hat[0, -1, 0, :, :].cpu().data.numpy()
        #    img2 = targets[0, -1, 0, :, :].cpu().data.numpy()
        #    ## Plot current prediction
        #    im1 = axes[0, 1].imshow(img1, cmap="RdBu_r")
        #    plt.colorbar(im1, ax=axes[0, 1])
        #    axes[0, 1].set_title("y_hat: last leadtime")
        #    ## Plot target
        #    im2 = axes[1, 1].imshow(img2, cmap="RdBu_r")
        #    plt.colorbar(im2, ax=axes[1, 1])
        #    axes[1, 1].set_title("target: last leadtime")
        #    plt.show()
        #self.count += 1

        # Computing using nn.MSELoss base class. This class must be instantiated via:
        # criterion = nn.MSELoss(reduction="none")
        loss = super().forward(y_hat, targets)

        # Computing here, in the nn.Module derived class
        # loss = (
        #             ( ( y_hat.movedim(-1, 1) - targets.movedim(-1, 1) )*100 )**2
        #         )*sample_weights.movedim(-1, 1)
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
        y_hat = inputs.movedim(-2, 1).movedim(-1, 1)
        targets = targets.movedim(-1, 1).movedim(-1, 1)
        sample_weights = sample_weights.movedim(-1, 1).movedim(-1, 1)

        y_hat = y_hat * sample_weights
        targets = targets * sample_weights

        return super().forward(y_hat, targets).mean()

        # # Computing using nn.HuberLoss base class. This class must be instantiated via:
        # # criterion = nn.HuberLoss(reduction="none")
        # loss = super().forward(y_hat, targets)

        # return loss.mean()
