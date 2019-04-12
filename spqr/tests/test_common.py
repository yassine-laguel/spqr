import pytest

from sklearn.utils.estimator_checks import check_estimator

from spqr import TemplateEstimator
from spqr import TemplateClassifier
from spqr import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
