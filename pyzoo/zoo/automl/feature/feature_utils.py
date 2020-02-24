from zoo.automl.feature.base import BaseEstimator
from sklearn.exceptions import NotFittedError


def check_is_fitted(estimator, attributes):
    """
    check if the estimator is fitted by attributes
    :param estimator:
    :param attributes: string or a list/tuple of strings
    :return:
    """
    msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
           "appropriate arguments before using this method.")

    if not isinstance(estimator, BaseEstimator):
        raise TypeError("%s is not an estimator instance." % estimator)

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})
