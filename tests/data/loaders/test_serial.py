# from src.canari_ml.data.loaders import SerialLoader
# import pytest
import pandas as pd

# import numpy as np
import xarray as xr

# from datetime import datetime
from dateutil.relativedelta import relativedelta

from src.canari_ml.data.loaders.serial import get_date_indices


def test_get_date_indices_days() -> None:
    """
    Test getting time indicies for input and output prediction data for daily
    forecast.

    Notes: I've gone heavy on comments even for the obvious parts
        for my own benefit.
    """
    # Create a facsimile xarray dataset with time values
    start_date = pd.Timestamp("2020-01-01")
    times = [start_date + pd.Timedelta(days=i) for i in range(10)]
    var_ds = xr.Dataset({"time": ("time", times)})
    var_ds = var_ds.assign_coords(time=("time", times))

    # Given a forecast initialisation date:
    forecast_date = times[3]  # 2020-01-04
    # Wanting to forecast for this number of days
    n_forecast_steps = 4
    # Assuming daily forecasting
    relative_attr = "days"

    forecast_base_idx, forecast_idxs, forecast_steps_gen = get_date_indices(
        forecast_date, var_ds, n_forecast_steps, relative_attr
    )

    # Check forecast_base_idx
    assert forecast_base_idx == 3

    # Check the indices for the expected output prediction.
    # This should be the forecast init date + n_forecast_steps - 1
    assert forecast_idxs == [3, 4, 5, 6]

    # Check forecast_steps generator matches the expected dates
    expected_dates = [
        forecast_date + relativedelta(**{relative_attr: i})
        for i in range(n_forecast_steps)
    ]
    prediction_dates = list(forecast_steps_gen)
    assert prediction_dates == expected_dates


def test_get_date_indices_months() -> None:
    """
    Test `get_date_indices` for monthly stepping.
    """
    # Create a facsimile xarray dataset with monthly time values
    start_date = pd.Timestamp("2020-01-01")
    times = [start_date + pd.DateOffset(months=i) for i in range(6)]
    var_ds = xr.Dataset({"time": ("time", times)})
    var_ds = var_ds.assign_coords(time=("time", times))

    # Given a forecast initialisation date:
    forecast_date = times[2]  # 2020-03-01
    n_forecast_steps = 3
    relative_attr = "months"

    forecast_base_idx, forecast_idxs, forecast_steps_gen = get_date_indices(
        forecast_date, var_ds, n_forecast_steps, relative_attr
    )

    # Check forecast_base_idx
    assert forecast_base_idx == 2

    # Check the indices for the expected output prediction.
    assert forecast_idxs == [2, 3, 4]

    # Check forecast_steps generator matches the expected dates
    expected_dates = [
        forecast_date + relativedelta(**{relative_attr: i})
        for i in range(n_forecast_steps)
    ]
    prediction_dates = list(forecast_steps_gen)
    assert prediction_dates == expected_dates
