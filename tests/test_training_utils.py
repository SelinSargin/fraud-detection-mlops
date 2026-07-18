import pytest

from src.training_utils import calculate_threshold_metrics

def test_calculate_threshold_metrics():

    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.4, 0.35, 0.8]


    metrics = calculate_threshold_metrics(
        y_true=y_true,
        y_proba=y_proba,
        threshold=0.50

    )

    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1

    assert metrics["precision"] == pytest.approx(1.0)
                                                 
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx( 2/3)
    

