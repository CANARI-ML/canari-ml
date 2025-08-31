import pytest
import torch
from canari_ml.models.metrics import MAE, MSE, RMSE


@pytest.fixture
def test_data():
    predictions = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    targets = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    sample_weight = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    return predictions, targets, sample_weight


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metric_cls, expected",
    [
        (MAE, torch.tensor(1.0)),
        (MSE, torch.tensor(1.0)),
        (RMSE, torch.tensor(1.0)),
    ],
)
def test_metrics_basic(metric_cls, expected, test_data):
    preds, targets, weights = test_data
    metric = metric_cls()
    metric.update(preds, targets, weights)
    result = metric.compute()
    assert torch.isclose(result, expected), f"{metric_cls.__name__} failed"
