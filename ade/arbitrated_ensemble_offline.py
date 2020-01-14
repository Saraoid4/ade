#!/usr/bin/env python
__author__ = "Dihia BOULEGANE"
__copyright__ = ""
__credits__ = ["Dihia BOULEGANE"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Dihia BOULEGANE"
__email__ = "dihia.boulegane@telecom-paristech.fr"
__status__ = "Development"


from skmultiflow.utils import *
from utils.functions import *
from utils.metrics import *

from ade.arbitrated_ensemble import ArbitratedEnsemble


class ArbitratedEnsembleOffline(ArbitratedEnsemble):
    """ ArbitratedEnsembleOffline

    This model implements an ArbitratedEnsemble with an offline training phase on both meta and base level

    This approach has two phases:
        - Offline phase: We use batch of instances to pre-train meta and base models before using an arbitrated policy.
                         During his phase, the final prediction for a given instance is a simple average of all
                         base-models' outputs
        - Online phase:
                        When the number of instances seen in the stream is equal to the specified offline batch size, we
                        switch to the online phase where the behaviour is the one described in the ArbitratedEnsemble
                        model

    Parameters
    ----------

    meta_models: Array-like  of  StreamModel
        Pool of StreamModel where each model learns to predict the error rate of its base counter part

    base_models:
        Pool of StreamModel where each model learns to predict values of the stream

    offline_batch_size: int
        number of examples that must be seen before starting predicting at the meta-level

    meta_error_metric: String
        Measure that will be used on the meta level to estimate the error of the base models. Could be MSE or MAE or
        any other  measure to quantify models' error on a given instance.

    """

    def __init__(self, meta_models, base_models, pretrain_size, meta_error_metric='MSE'):

        super().__init__(meta_models=meta_models, base_models=base_models, meta_error_metric=meta_error_metric,
                         pretrain_size=pretrain_size)

        self.examples_seen = 0

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
        super().partial_fit(X, y)
        # Update number of examples seen
        r, _ = get_dimensions(X)
        self.examples_seen += r

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
        # TODO: change setting of offline phase

        if self.examples_seen <= self.pretrain_size:
            # We are in the offline phase ==> final prediction is equal to simple average of all base models
            for i in range(r):
                base_predictions = np.array([m.predict([X[i]])[0] for m in self.base_models]).reshape(1, -1)
                final_prediction = get_aggregated_result(base_predictions, weights=None, method='simple_average')
                predictions.append(final_prediction)
        else:
            # We are in the online phase, use arbitrating between base models
            for i in range(r):
                meta_predictions = [z.predict([X[i]])[0] for z in self.meta_models]
                weights = weight_predictions(meta_predictions)
                base_predictions = np.array([m.predict([X[i]])[0] for m in self.base_models]).reshape(1, -1)
                final_prediction = get_aggregated_result(base_predictions, weights, method='weighted_average')
                predictions.append(final_prediction)

        return predictions

    def predict_proba(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def reset(self):
        self.meta_models = []
        self.base_models = []

    def get_class_type(self):
        return 'ArbitratedEnsemble'

    def get_info(self):
        return 'Array BaseStreamModel: base_models: ' + str(self.base_models) + \
               '\n Array MetaStreamModel: base_models: ' + str(self.meta_models)


