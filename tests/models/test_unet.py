import pytest

import torch

from canari_ml.models.models import UNet


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
