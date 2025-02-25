import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_wrapper import LitUNet

# Suppress the specific UserWarning from PyTorch
warnings.filterwarnings(
    "ignore", message="Using padding='same' with even kernel lengths and odd dilation"
)


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        input_channels,
        filter_size=3,
        n_filters_factor=1,
        lead_time=7,
        n_output_classes=1,
        **kwargs,
    ):
        super(UNet, self).__init__()

        self.input_channels = input_channels
        self.filter_size = filter_size
        self.n_filters_factor = n_filters_factor
        self.lead_time = lead_time
        self.n_output_classes = n_output_classes

        start_out_channels = 64
        reduced_channels = int(start_out_channels * n_filters_factor)

        reduced_channels = int(start_out_channels * n_filters_factor)
        channels = {
            start_out_channels * 2**pow: reduced_channels * 2**pow for pow in range(4)
        }

        # Encoder
        self.conv1 = self.conv_block(input_channels, channels[64])
        self.conv2 = self.conv_block(channels[64], channels[128])
        self.conv3 = self.conv_block(channels[128], channels[256])
        self.conv4 = self.conv_block(channels[256], channels[256])

        # Bottleneck
        self.conv5 = self.bottleneck_block(channels[256], channels[512])

        # Decoder
        self.up6 = self.upconv_block(channels[512], channels[256])
        self.up7 = self.upconv_block(channels[256], channels[256])
        self.up8 = self.upconv_block(channels[256], channels[128])
        self.up9 = self.upconv_block(channels[128], channels[64])

        self.up6b = self.conv_block(channels[512], channels[256])
        self.up7b = self.conv_block(channels[512], channels[256])
        self.up8b = self.conv_block(channels[256], channels[128])
        self.up9b = self.conv_block(channels[128], channels[64], final=True)

        # Final layer
        self.final_layer = nn.Conv2d(
            channels[64], lead_time, kernel_size=1, padding="same"
        )

    def forward(self, x):
        # transpose from shape (b, h, w, c) to (b, c, h, w) for pytorch conv2d layers
        x = torch.movedim(x, -1, 1)  # move c from last to second dim

        # Encoder
        bn1 = self.conv1(x)
        conv1 = F.max_pool2d(bn1, kernel_size=2)
        bn2 = self.conv2(conv1)
        conv2 = F.max_pool2d(bn2, kernel_size=2)
        bn3 = self.conv3(conv2)
        conv3 = F.max_pool2d(bn3, kernel_size=2)
        bn4 = self.conv4(conv3)
        conv4 = F.max_pool2d(bn4, kernel_size=2)

        # Bottleneck
        bn5 = self.conv5(conv4)

        # Decoder
        up6 = self.up6b(torch.cat([bn4, self.up6(bn5)], dim=1))
        up7 = self.up7b(torch.cat([bn3, self.up7(up6)], dim=1))
        up8 = self.up8b(torch.cat([bn2, self.up8(up7)], dim=1))
        up9 = self.up9b(torch.cat([bn1, self.up9(up8)], dim=1))

        # Final layer
        output = self.final_layer(up9)

        # Convert raw logits to result
        # y_hat = torch.sigmoid(output)
        # For CANARI, output will be unbounded floats (i.e., not limited to 0-1)
        y_hat = output

        # transpose from shape (b, c, h, w) back to (b, h, w, c) to align with training data
        y_hat = torch.movedim(y_hat, 1, -1)  # move c from second to final dim

        b, h, w, c = y_hat.shape

        # unpack c=classes*months dimension into classes, months as separate dimensions
        y_hat = y_hat.reshape((b, h, w, self.n_output_classes, self.lead_time))

        return y_hat

    def conv_block(self, in_channels, out_channels, final=False):
        block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=self.filter_size, padding="same"
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=self.filter_size, padding="same"
            ),
            nn.ReLU(inplace=True),
        )
        if not final:
            batch_norm = nn.Sequential(
                nn.BatchNorm2d(num_features=out_channels),
            )
            return nn.Sequential().extend(block).extend(batch_norm)
        else:
            final_block = nn.Sequential(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=self.filter_size,
                    padding="same",
                ),
                nn.ReLU(inplace=True),
            )
            return nn.Sequential().extend(block).extend(final_block)

    def bottleneck_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=self.filter_size, padding="same"
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=self.filter_size, padding="same"
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            Interpolate(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding="same"),
            nn.ReLU(inplace=True),
        )


def unet_batchnorm(
    input_shape: object,
    loss: object,
    metrics: object,
    learning_rate: float = 1e-4,
    custom_optimizer: object = None,
    filter_size: float = 3,
    n_filters_factor: float = 1,
    lead_time: int = 1,
) -> object:
    # construct unet
    model = UNet(
        input_channels=input_shape[-1],
        filter_size=filter_size,
        n_filters_factor=n_filters_factor,
        lead_time=lead_time,
    )

    # criterion = BCELoss(reduction="none")
    # criterion = L1Loss(reduction="none")
    # criterion = MSELoss(reduction="none")

    # configure PyTorch Lightning module
    lit_module = LitUNet(
        model=model,
        criterion=loss,
        learning_rate=learning_rate,
        metrics=metrics,
        enable_leadtime_metrics=False,
    )

    return lit_module
