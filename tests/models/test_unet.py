import pytest

import torch

from canari_ml.models.models import UNet


@pytest.fixture
def unet():
    return UNet(
        input_channels=3,
        filter_size=3,
        n_filters_factor=4,
        lead_time=7,
        n_output_classes=1,
        dropout_probability=0.3,
    )


@pytest.mark.unet
@pytest.mark.parametrize(
    (
        "input_channels,"
        "filter_size,"
        "n_filters_factor,"
        "lead_time,"
        "n_output_classes,"
        "dropout_probability"
    ),
    [(3, 3, 1, 7, 1, 0.3), (12, 3, 5, 10, 3, 1)],
)
def test_unet_init(
    input_channels,
    filter_size,
    n_filters_factor,
    lead_time,
    n_output_classes,
    dropout_probability,
):
    unet = UNet(
        input_channels=input_channels,
        filter_size=filter_size,
        n_filters_factor=n_filters_factor,
        lead_time=lead_time,
        n_output_classes=n_output_classes,
        dropout_probability=dropout_probability,
    )
    assert unet.input_channels == input_channels, (
        f"Expected {unet.input_channels} to be {input_channels}"
    )
    assert unet.filter_size == filter_size, (
        f"Expected {unet.filter_size} to be {filter_size}"
    )
    assert unet.n_filters_factor == n_filters_factor, (
        f"Expected {unet.n_filters_factor} to be {n_filters_factor}"
    )
    assert unet.lead_time == lead_time, f"Expected {unet.lead_time} to be {lead_time}"
    assert unet.n_output_classes == n_output_classes, (
        f"Expected {unet.n_output_classes} to be {n_output_classes}"
    )
    assert unet.dropout_probability == dropout_probability, (
        f"Expected {unet.dropout_probability} to be {dropout_probability}"
    )


# EASE2 grid resolution ref:
# https://nsidc.org/data/user-resources/help-center/guide-ease-grids
@pytest.mark.unet
@pytest.mark.parametrize(
    "input_dims, expected_dims",
    [
        # Input shape: (batch_size, channels, height, width)
        # Output shape: (batch_size, unet.n_output_classes, height, width, unet.lead_time)
        ((1, 3, 500, 500), (1, 1, 500, 500, 7)),
        ((8, 3, 500, 500), (8, 1, 500, 500, 7)),  # EASE2 36km2 grid (EASE2_N36km)
        ((4, 3, 720, 720), (4, 1, 720, 720, 7)),  # EASE2 25km2 grid (EASE2_N25km)
        ((4, 3, 201, 201), (4, 1, 201, 201, 7)),  # Odd img dims
        ((4, 3, 201, 500), (4, 1, 201, 500, 7)),  # Non-square img dims
        ((7, 3, 31, 31), (7, 1, 31, 31, 7)),  # Small dims
    ],
)
def test_unet_forward_pass(unet, input_dims, expected_dims):
    assert unet.n_output_classes == 1

    # Random tensor from normal dist w/ mean 0, var 1
    x = torch.randn(input_dims)
    y_hat = unet.forward(x)

    # Expected output tensor of shape
    assert y_hat.shape == expected_dims, f"Expected {y_hat.shape} to be {expected_dims}"
