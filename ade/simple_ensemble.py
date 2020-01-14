#!/usr/bin/env python
__author__ = "Dihia BOULEGANE"
__copyright__ = ""
__credits__ = ["Dihia BOULEGANE"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Dihia BOULEGANE"
__email__ = "dihia.boulegane@telecom-paristech.fr"
__status__ = "Development"

from skmultiflow.core.base import StreamModel
from utils.functions import *
from skmultiflow.utils import *


class SimpleEnsemble(StreamModel):
    """ ArbitratedEnsemble

    The arbitrated architecture implemented in this class is designed to do Dynamic Ensemble Selection using $
    meta-learning to predict future values of a stream.

    Parameters
    ----------

    meta_models: Array-like  of  StreamModel
        Pool of StreamModel where each model learns to predict the error rate of its base counter part

    base_models:
        Pool of StreamModel where each model learns to predict values of the stream

    """
    

    def __init__(self, base_models):
        """

        Parameters
        ----------
        meta_models
        base_models
        """
        super().__init__()
        self.base_models = base_models

        # TODO : Handle output files


    def fit(self, X, y, classes=None, weight=None):
        """ Fit the model under the batch setting (Not implemented in this case).

        Parameters
        ----------
        X:  Numpy.ndarray of shape (n_samples, n_features)
            The array of samples used to fit the model.
        y: Array-like
            An array-like with the labels of all samples in X.
        classes: Array-like, optional (default=None)
            Contains all possible labels. Applicability varies depending on the algorithm.
        weight: Array-like, optional (default=None)
            Instance weight. If not provided, uniform weights are assumed.
            Applicability varies depending on the algorithm.

        Returns
        -------
        self
        """
        raise NotImplementedError

    def partial_fit(self, X, y, weight=None):
        """ Partial (incremental) fit the model under the stream learning setting

        Parameters
        ----------
        X:  Numpy.ndarray of shape (n_samples, n_features)
            The array of samples used to fit the model.
        y: Array-like
            An array-like with the labels of all samples in X.
        weight: Array-like, optional (default=None)
            Instance weight. If not provided, uniform weights are assumed.
            Applicability varies depending on the algorithm.

        Returns
        -------
        self
        """

        # TODO: handle new concepts
        for m in self.base_models:
            # Incremental training of all  base-models
            m.partial_fit(X, y, weight)

    def predict(self, X):

        """ Predicts target using the arbitrated ensemeble model.

        Parameters
        ----------
        X : Numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict.

        Returns
        -------
        list of all predicted for  samples in X.

        """
        r, _ = get_dimensions(X)
        predictions = []

        for i in range(r):
            base = [m.predict([X[i]])[0] for m in self.base_models]
            final_prediction = get_aggregated_result(predictions=base, weights=None,  method='simple_average')
            predictions.append(final_prediction)

        return predictions

    def predict_proba(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def reset(self):
        self.base_models = []

    def get_class_type(self):
        return 'SimpleEnsemble'

    def get_info(self):
        info = '{}:'.format(type(self).__name__)
        info += ' - ensemble_size: {}'.format(len(self.base_models))
        return info

    def get_model_name(self):
        return 'ADE_NAIVE_SIZE_' + str(len(self.base_models))



