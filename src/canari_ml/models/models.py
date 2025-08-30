import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress the specific UserWarning from PyTorch on performance impact due
# to padding="same"
warnings.filterwarnings(
    "ignore", message="Using padding='same' with even kernel lengths and odd dilation"
)

def get_padding(x: torch.Tensor, stride: int = 16) -> tuple[int, int, int, int]:
    """Calculate padding for the input tensor to make its height and width multiples
    of the given stride.

    Args:
        x: Input tensor of shape (..., H, W).
        stride: Stride value to align dimensions to. Defaults to 16.

    Returns:
        Four integers representing the padding amounts in the format
        (left, right, top, bottom).

    # Notes:
    #     Reference this URL:
    #     https://stackoverflow.com/questions/66028743/how-to-handle-odd-resolutions-in-unet-architecture-pytorch
    """
    h, w = x.shape[-2:]

    # Calculate new dimensions that are multiples of the stride
    new_h = ((h + stride - 1) // stride) * stride
    new_w = ((w + stride - 1) // stride) * stride

    # Compute padding amounts for height and width
    pad_h = new_h - h
    pad_w = new_w - w

    # Divide padding symmetrically
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

    return (pad_left, pad_right, pad_top, pad_bottom)

def apply_padding(x: torch.Tensor, padding: tuple) -> torch.Tensor:
    """Apply zero-padding to the input tensor.

    Args:
        x: Input tensor to be padded.
        padding: Tuple specifying padding amounts for each dimension in the format
            (left, right, top, bottom).

    Returns:
        Tensor with padding applied.

    Notes:
        Using Zero padding across boundaries.
        Other options: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
    """
    out = F.pad(x, padding, "constant", value=0)
    return out

def undo_padding(x: torch.Tensor, padding: tuple) -> torch.Tensor:
    """Remove padding from the input tensor based on the specified padding
    dimensions.

    Args:
        x: Input tensor that was padded using `apply_padding`.
        padding: The padding tuple returned by `get_padding` in the format
            (left, right, top, bottom).

    Returns:
        The input tensor with padding cropped out.
    """
    pad_left, pad_right, pad_top, pad_bottom = padding
    if pad_top + pad_bottom > 0:
        x = x[:, :, pad_top:-pad_bottom, :]
    if pad_left + pad_right > 0:
        x = x[:, :, :, pad_left:-pad_right]
    return x

class Interpolate(nn.Module):
    """Module for upsampling using interpolation.

    Attributes:
        scale_factor: Scaling factor for upsampling.
        mode: Interpolation mode (e.g., 'nearest', 'bilinear').
    """
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.interp = F.interpolate
        self.scale_factor: float = scale_factor
        self.mode: str = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply interpolation to input tensor.

        Args:
            x: Input tensor.

        Returns:
            Upsampled tensor.
        """
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class UNet(nn.Module):
    """U-Net architecture for CANARI-ML forecasting.

    This implementation supports dynamic input sizes through padding and reshapes
    the output to include a lead time dimension.

    Attributes:
        input_channels: Number of input channels.
        filter_size: Size of convolutional filters (Default is 3).
        n_filters_factor: Factor to scale the number of filters in each layer.
        lead_time: Number of forecast or output time steps.
        n_output_classes: Number of output channels per time step.
        dropout_probability: Dropout probability used in encoder and decoder.
    """
    def __init__(
        self,
        input_channels: int,
        filter_size: int = 3,
        n_filters_factor: float = 1,
        lead_time=7,
        n_output_classes: int = 1,
        dropout_probability: float = 0.3,
        **kwargs,
    ):
        super(UNet, self).__init__()

        self.input_channels = input_channels
        self.filter_size = filter_size
        self.n_filters_factor = n_filters_factor
        self.lead_time = lead_time
        self.n_output_classes = n_output_classes
        self.dropout_probability = dropout_probability

        start_out_channels = 64
        reduced_channels = int(start_out_channels * n_filters_factor)

        reduced_channels = int(start_out_channels * n_filters_factor)
        channels = {
            start_out_channels * 2**pow: reduced_channels * 2**pow for pow in range(4)
        }

        self.cached_padding: tuple[int, int, int, int] | None = None

        self.dropout = nn.Dropout2d(dropout_probability)

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
            channels[64],
            self.n_output_classes * self.lead_time,
            kernel_size=1,
            padding="same"
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the U-Net.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C_out, H, W, T), where
                C_out = n_output_classes and T = lead_time.
        """
        # Number of `max_pool2d` steps in Encoder layer.
        # i.e., dimension halved, since it must later be doubled, else, will
        # be a dimension mismatch in the decoder layer,
        # e.g. 65 (131/2) vs 64 (32x2).
        # This determines number of times the input dimension must be perfectly
        # divisible by 2, and if not, must be padded for this architecture.
        n_max_pool = 4
        max_pool_kernel_size = 2
        stride = max_pool_kernel_size**n_max_pool

        h, w = x.shape[-2:]
        # Pad data if the input image dimensions is not a multiple of stride
        if h % stride or w % stride:
            if self.cached_padding is None:
                self.cached_padding = get_padding(x, stride=stride)
                self.cached_input_shape = (h, w)

            x = apply_padding(x, self.cached_padding)
        else:
            self.cached_padding = None

        # Encoder
        bn1 = self.conv1(x)
        conv1 = F.max_pool2d(bn1, kernel_size=max_pool_kernel_size)
        conv1 = self.dropout(conv1)
        bn2 = self.conv2(conv1)
        conv2 = F.max_pool2d(bn2, kernel_size=max_pool_kernel_size)
        conv2 = self.dropout(conv2)
        bn3 = self.conv3(conv2)
        conv3 = F.max_pool2d(bn3, kernel_size=max_pool_kernel_size)
        conv3 = self.dropout(conv3)
        bn4 = self.conv4(conv3)
        conv4 = F.max_pool2d(bn4, kernel_size=max_pool_kernel_size)
        conv4 = self.dropout(conv4)

        # Bottleneck
        bn5 = self.conv5(conv4)

        # Decoder
        up6 = self.up6b(torch.cat([bn4, self.up6(bn5)], dim=1))
        up6 = self.dropout(up6)
        up7 = self.up7b(torch.cat([bn3, self.up7(up6)], dim=1))
        up7 = self.dropout(up7)
        up8 = self.up8b(torch.cat([bn2, self.up8(up7)], dim=1))
        up8 = self.dropout(up8)
        up9 = self.up9b(torch.cat([bn1, self.up9(up8)], dim=1))

        # Final layer
        output = self.final_layer(up9)

        # Undo padding of the network output to recover the original input shape
        if self.cached_padding:
           output = undo_padding(output, self.cached_padding)

        # Convert raw logits to result
        # Can do without since we're working with regression w/ continuous values
        # y_hat = torch.sigmoid(output)
        y_hat = output

        b, c, h, w = y_hat.shape

        # unpack c=classes*days dimension into classes, days as separate dimensions
        # Reshape to: (b, n_output_classes, h, w, lead_time)
        y_hat = y_hat.reshape(b, self.n_output_classes, self.lead_time, h, w) 
        y_hat = y_hat.permute(0, 1, 3, 4, 2).contiguous()  # (b, classes, h, w, time)

        return y_hat

    def conv_block(
        self, in_channels: int, out_channels: int, final: bool = False
    ) -> nn.Sequential:
        """Builds a standard convolutional block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            final (optional): If True, adds a third conv + ReLU instead of batch norm.
                Defaults to False.

        Returns:
            Sequential container with conv layers.
        """
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
            block.append(batch_norm)
        else:
            block.extend([
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=self.filter_size,
                    padding="same",
                ),
                nn.ReLU(inplace=True),
            ])
        return nn.Sequential(*block)


    def bottleneck_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Builds the bottleneck block between encoder and decoder.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.

        Returns:
            Bottleneck block with conv, ReLU, and batch norm.
        """
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

    def upconv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Builds an upsampling block using interpolation followed by convolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.

        Returns:
            Upsampling block.
        """
        return nn.Sequential(
            Interpolate(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding="same"),
            # Upscale the input by a factor of 2 (using instead of
            # torch.nn.functional.interpolator or nn.Upsample with Conv2d)
            # nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            # nn.ReLU(inplace=True),
        )
