#!/usr/bin/env python

__author__ = "Dihia BOULEGANE"
__copyright__ = ""
__credits__ = ["Dihia BOULEGANE"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Dihia BOULEGANE"
__email__ = "dihia.boulegane@telecom-paristech.fr"
__status__ = "Development"

import numpy as np
import sys
from skmultiflow.utils import *
from utils.functions import *
from utils.metrics import *
from core.queue import Queue

from ade.abstract_arbitrated_ensemble import AbstractArbitratedEnsemble


class ArbitratedEnsembleAbstainingProb(AbstractArbitratedEnsemble):
    """ ArbitratedEnsembleAbstaining

    The arbitrated architecture implemented in this class is designed to do Dynamic Ensemble Selection using $
    meta-learning to predict future values of a stream. It's based on an abstaining policy to select the most confident
    base-models only to contribute to the final output and weight them inversely to the estimated error by the the
    meta-models.

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


    """

    # TODO: handle concept drift ad new models
    def __init__(self, meta_models, base_models, meta_error_metric='MSE', conf_random=True):

        super().__init__(meta_models=meta_models, base_models=base_models, meta_error_metric=meta_error_metric)

        # TODO: Add randomized variable on experts selection
        self.conf_random = conf_random
        self.first_run = True
        self.previous_predictions = Queue()

        # TODO refactor creation of Normal Random Law for random selection

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
        r, _ = get_dimensions(X)
        predictions = []
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
                weights = weight_predictions(meta_predictions)

            try:
                final_prediction = get_aggregated_result(base_predictions, weights, method='weighted_average')
                predictions.append(final_prediction)
            except Exception as exc:
                raise exc

        self.previous_predictions.enqueue(predictions)
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
            # TODO: handle negative errors predicted
            predicted_error = z.predict(X)[0]

            # random bernoulli on confidence
            if self.conf_random:
                random_conf = np.random.binomial(1, z.get_confidence_score(), 1)
            else:
                random_conf = 1

            # random bernoulli inverse predicted error
            p = 1-np.abs(predicted_error)/100
            if p > 1 or p < 0:
                random_error = np.random.binomial(1, sys.float_info.epsilon, 1)
            else:
                random_error = np.random.binomial(1, p, 1)
            if random_conf * random_error:
                # select base model
                sub_meta_models.append(z)
                sub_base_models.append(self.base_models[i])
                sub_meta_predictions.append(predicted_error)
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
        return 'ArbitratedEnsembleAbstainProb'

    def get_info(self):
        return 'Array BaseStreamModel: base_models: ' + str(self.base_models) + \
               '\n Array MetaStreamModel: meta_models: ' + str(self.meta_models)

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

        if self.first_run:
            # Train on all instances before predicting
            self.first_run = False
            for z, m in zip(self.meta_models, self.base_models):
                m.partial_fit(X, y)
                predictions = m.predict(X)
                real_errors = np.array([get_predictions_errors(real_y, estimated_y, self.meta_error_metric)
                                        for real_y, estimated_y in zip(y, predictions)])
                # Incremental learning of meta-models
                z.partial_fit(X, real_errors, weight)

        else:
            r, _ = get_dimensions(X)

            for z, m in zip(self.meta_models, self.base_models):
                # Incremental training of base base models
                # Getting base models predictions
                predictions = m.predict(X)
                # Computing real incurred errors
                real_errors = np.array([get_predictions_errors(real_y, estimated_y, self.meta_error_metric)
                                        for real_y, estimated_y in zip(y, predictions)])

                # Incremental learning of meta-models
                z.partial_fit(X, real_errors, weight)
                m.partial_fit(X, y, weight)

            predictions = [m.predict(X) for m in self.base_models]
            # TODO: deal in case no real experts to add new models
            real_errors_concept_drift = np.array([get_predictions_errors(real_y, estimated_y, self.meta_error_metric)
                                                  for real_y, estimated_y in zip(y, predictions)])






