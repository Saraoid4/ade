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
from skmultiflow.utils import *
from utils.functions import *
from utils.metrics import *


class ArbitratedEnsembleAbstainingOffline(StreamModel):
    """ ArbitratedEnsemble

    The arbitrated architecture implemented in this class is designed to do Dynamic Ensemble Selection using $
    meta-learning to predict future values of a stream.

    Parameters
    ----------

    meta_models: Array-like  of  StreamModel
        Pool of StreamModel where each model learns to predict the error rate of its base counter part
    base_models:
        Pool of StreamModel where each model learns to predict values of the stream
    competence_threshold: float
        Maximum error tolerated from a model to contribute in the ensemble
    beta: int
        minimum size of outliers buffer to trigger a new regime
    abstain_learning: boolean
        if True, update  real experts  only on the base level

    """

    # TODO: handle all data structures in class
    def __init__(self, meta_models, base_models, offline_batch_size, competence_threshold=100, beta=100, abstain_learning=False):

        super().__init__()
        self.meta_models = meta_models
        self.base_models = base_models
        self.offline_batch_size = offline_batch_size
        self.examples_seen = 0
        self.competence_threshold = competence_threshold
        self.beta = beta
        self.abstain_learning = abstain_learning

        self._outliers_buffer = []

    # TODO: check fit function if needed or delete
    def fit(self, X, y, classes=None, weight=None):
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
        self._fit_all(X, y)

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
        # TODO: handle outliers buffer

        r, _ = get_dimensions(X)
        predictions = []
        if self.examples_seen > self.offline_batch_size:
            # We are in the online phase ==> do dynamic selection
            for i in range(r):

                # TODO : Handle when first predictions are negative add some random predictions
                meta_models, base_models, meta_predictions, base_predictions = self._select_experts(np.array([X[i]]))

                if len(base_models) > 0:
                    """ Sub set of experts found"""
                    base_predictions = np.array(base_predictions).reshape(1, -1)
                    weights = weight_predictions(meta_predictions)

                else:
                    meta_predictions = [z.predict([X[i]])[0] for z in self.meta_models]
                    base_predictions = np.array([m.predict([X[i]])[0] for m in self.base_models]).reshape(1, -1)
                    weight = weight(meta_predictions)

                final_prediction = get_aggregated_result(base_predictions, weights, method='weighted_average')
                predictions.append(final_prediction)
        else:
            # We are in the offline phase ==> Average all
            for i in range(r):
                base_predictions = np.array([m.predict([X[i]])[0] for m in self.base_models]).reshape(1, -1)
                final_prediction = get_aggregated_result(base_predictions, weights=None, method='simple_average')
                predictions.append(final_prediction)

        return predictions

    def _select_experts(self, X):
        """
        Selects subset of base models as experts based on the predictions of the meta models
        Parameters
        ----------
        X: Array-like
            Test instance on which will be based the selection of experts

        Returns
        -------
        # TODO: Document return values
        Tuple :
        """
        sub_meta_models = []
        sub_base_models = []
        sub_meta_predictions = []
        sub_base_predictions = []
        for i in range(len(self.meta_models)):
            z = self.meta_models[i]
            # TODO: recheck data structure for returned prediction
            predicted_error = z.predict(X)[0]
            if predicted_error <= self.competence_threshold:
                sub_meta_models.append(z)
                sub_base_models.append(self.base_models[i])
                sub_meta_predictions.append(predicted_error)
                # TODO : verify return structure
                sub_base_predictions.append(self.base_models[i].predict(X)[0])
        return sub_meta_models, sub_base_models, sub_meta_predictions, sub_base_predictions

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
               '\n Array BaseStreamModel: base_models: ' + str(self.base_models)

    def _fit_all(self, X, y, weight=None):
        """ performs a partial fit for all meta and base model

        Parameters
        ----------
        X:  Numpy.ndarray of shape (n_samples, n_features)
            The array of samples used to fit the model.
        y: Array-like
            An array-like with the labels of all samples in X.
        weight: Array-like, optional (default=None)
            Instance weight. If not provided, uniform weights are assumed.
            Applicability varies depending on the algorithm.

        """
        r, _ = get_dimensions(X)

        # TODO: abstain learning
        for z, m in zip(self.meta_models, self.base_models):
            # Incremental training of base base models
            m.partial_fit(X, y, weight)
            # Getting base models predictions
            predictions = m.predict(X)
            # Computing real incurred errors
            real_errors = np.array([compute_mse(real_y, estimated_y) for estimated_y, real_y in zip(predictions, y)])
            # Incremental learning of meta-models
            z.partial_fit(X, real_errors, weight)

        self.examples_seen += r

    def _get_real_experts(self, X, y):
        """

        Parameters
        ----------
        X: instance
        y: float

        Returns
        -------
        List of meta-models and base-models corresponding to real experts according to competence_threshold
        """
        base_experts = [m for m,z in zip(self.base_models, self.meta_models) if z.predict(np.array(X))
                            <= self.competence_threshold]

        return base_experts




