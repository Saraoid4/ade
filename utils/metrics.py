__author__ = "Dihia BOULEGANE"
__copyright__ = ""
__credits__ = ["Dihia BOULEGANE"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Dihia BOULEGANE"
__email__ = "dihia.boulegane@telecom-paristech.fr"
__status__ = "Development"

import numpy
import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

MSE = 'MSE'
RMSE = 'RMSE'
MAE = 'MAE'
MAPE = 'MAPE'
MASE = 'MASE'


def get_predictions_errors(y, estimated_y, metric=MSE):
    if metric == MSE:
        return compute_mse(y, estimated_y)
    elif metric == MAE:
        return compute_mae(y, estimated_y)
    elif metric == MAPE:
        return compute_mape(y, estimated_y)
    elif metric == RMSE:
        return compute_rmse(y, estimated_y)
    else:
        raise ValueError('Unknowm metric ', metric)


def compute_mse(y, estimated_y):
    """ MSE

    Computes the Mean Square Error

    Parameters
    ----------
    y: array-like or float
        Real value of target y
    estimated_y: array-like or float
        Predicted value of the target y

    Returns
    -------
        Mean Square Error
    """
    y, estimated_y = check_and_format_input(y, estimated_y)
    if mean_squared_error(y, estimated_y) < 0:
        print("MSE Negative error")
    return mean_squared_error(y, estimated_y)


def compute_rmse(y, estimated_y):
    return math.sqrt(compute_mse(y, estimated_y))


def compute_mae(y, estimated_y):
    """ MAE

    Parameters
    ----------
    y: array-like or float
    estimated_y: array-like or float

    Returns
    -------
        Mean Absolute Error
    """
    y, estimated_y = check_and_format_input(y, estimated_y)
    return mean_absolute_error(y, estimated_y)


def MASE(y, estimated_y):
    """

    Parameters
    ----------
    y
    estimated_y

    Returns
    -------

    """
    # TODO: Implement the Mean Absolute Scaled Error
    raise NotImplementedError


def compute_mape(y, estimated_y):
    """

    Parameters
    ----------
    y: array-like or float
        array of actual values of the target
    estimated_y: array-like or float
        array of the corresponding estimated values of y by a forecaster

    Returns
    -------
        Symmetric Mean Absolute Percentage Error as float
    """

    # TODO: Implement sMAPE to Deal with zeros
    y_true, y_pred = np.array(y), np.array(estimated_y)

    if not np.any(y_true) and not np.any(estimated_y):
        mape = 0
    else:
        y_true, y_pred = np.array(y), np.array(estimated_y)
        # TODO: writing could be simplified
        # Remove 2 and 100 to make error in [0,1]
        # mape = np.mean(2 * np.abs((y_pred - y_true)) / (np.abs(y_true) + np.abs(y_pred))) * 100
        mape = np.mean(np.abs((y_pred - y_true)) / (np.abs(y_true) + np.abs(y_pred)))

    return mape


def check_and_format_input(y, estimated_y):
    if hasattr(y, "__len__") and  hasattr(estimated_y, "__len__"):
        if len(y) != len(estimated_y) :
            raise ValueError('shapes' + numpy.shape(y) + 'and' + numpy.shape(estimated_y) + 'not aligned:' +
                              len(y) + '(dim' + numpy.ndim(y) + ') !=' + len(estimated_y) +
                              '(dim' + numpy.ndim(estimated_y)+ ')')

    else:
        try:
            y = [float(y)]
            estimated_y = [float(estimated_y)]

        except:
            raise TypeError('Check variables both numbers or vectors of same length')

    return y, estimated_y
