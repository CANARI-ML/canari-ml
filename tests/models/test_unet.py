import pytest

import torch

from canari_ml.models.models import UNet, get_padding, apply_padding, undo_padding


@pytest.fixture
def unet():
    return UNet(
        input_channels=3,
        filter_size=3,
        n_filters_factor=4,
        lead_time=3,
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
        # # Input shape: (batch_size, channels, height, width)
        # # Output shape: (batch_size, unet.n_output_classes, height, width, unet.lead_time)
        # ((8, 3, 500, 500), (8, 1, 500, 500, 3)),  # EASE2 36km2 grid (EASE2_N36km)
        # ((4, 3, 720, 720), (4, 1, 720, 720, 3)),  # EASE2 25km2 grid (EASE2_N25km)
        ((1, 3, 25, 25), (1, 1, 25, 25, 3)),
        ((2, 3, 51, 51), (2, 1, 51, 51, 3)),  # Odd img dims
        ((2, 3, 51, 50), (2, 1, 51, 50, 3)),  # Non-square img dims
        ((7, 3, 31, 31), (7, 1, 31, 31, 3)),  # Small dims
    ],
)
def test_unet_forward_pass(unet, input_dims, expected_dims):
    assert unet.n_output_classes == 1

    # Random tensor from normal dist w/ mean 0, var 1
    x = torch.randn(input_dims)
    y_hat = unet.forward(x)

    # Expected output tensor of shape
    assert y_hat.shape == expected_dims, f"Expected {y_hat.shape} to be {expected_dims}"


@pytest.mark.unet
@pytest.mark.parametrize(
    "input_dims, expected_padding",
    [
        # Input shape: (batch_size, channels, height, width)
        ((2, 3, 500, 500), (6, 6, 6, 6)),  # 500x500 needs padding
        ((2, 3, 720, 720), (0, 0, 0, 0)),  # 720x720 doesn't need padding
    ],
)
def test_get_padding(input_dims, expected_padding):
    x = torch.randn(input_dims)
    padding = get_padding(x, stride=16)

    assert padding == expected_padding, "Expected {padding} to be {expected_padding}"

    # Check padding is a 4 length tuple
    assert isinstance(padding, tuple) and len(padding) == 4, (
        f"Expected padding as tuple of length 4, got {padding}"
    )

    # Padding must not be negative
    assert all(p >= 0 for p in padding), "Padding values should be non-negative"


@pytest.mark.unet
@pytest.mark.parametrize(
    "input_dims, padding, expected_padded_dims",
    [
        # Input shape: (batch_size, channels, height, width)
        ((2, 3, 500, 500), (6, 6, 6, 6), (2, 3, 512, 512)),  # 500x500 needs padding
        (
            (2, 3, 720, 720),
            (0, 0, 0, 0),
            (2, 3, 720, 720),
        ),  # 720x720 doesn't need padding
    ],
)
def test_apply_padding(input_dims, padding, expected_padded_dims):
    x = torch.randn(input_dims)
    padded_x = apply_padding(x, padding)

    # One way of checking if the padding has been applied correctly
    height, width = input_dims[2:]
    padded_height = height + padding[2] + padding[3]
    padded_width = width + padding[0] + padding[1]

    # Ensure padding has been applied (programmatically)
    assert padded_x.shape[2] == padded_height, (
        f"Expected padded height {padded_height} {padded_x.shape[2]}"
    )
    assert padded_x.shape[3] == padded_width, (
        f"Expected padded width {padded_width}, got {padded_x.shape[3]}"
    )

    # Checking the same in another way
    assert padded_x.shape == expected_padded_dims, (
        f"Expected padded dims of {padded_x.shape} to be {expected_padded_dims}"
    )


@pytest.mark.unet
@pytest.mark.parametrize(
    "input_dims_with_padding, padding, expected_dims",
    [
        # Input shape: (batch_size, channels, height, width)
        (
            (2, 3, 500 + 6 + 6, 500 + 6 + 6),
            (6, 6, 6, 6),  # 500x500 needs padding
            (2, 3, 500, 500),
        ),
        (
            (2, 3, 720 + 0 + 0, 720 + 0 + 0),
            (0, 0, 0, 0),  # 720x720 doesn't need padding
            (2, 3, 720, 720),
        ),
    ],
)
def test_undo_padding(input_dims_with_padding, padding, expected_dims):
    x = torch.randn(input_dims_with_padding)
    cropped_x = undo_padding(x, padding)

    # Ensure padding has been removed
    assert cropped_x.shape == expected_dims, (
        f"Expected unpadded dims of {cropped_x.shape} to be {expected_dims}"
    )
