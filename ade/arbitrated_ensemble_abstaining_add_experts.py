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

from skmultiflow.utils import *
from utils.functions import *
from utils.metrics import *
from core.queue import Queue

from ade.abstract_arbitrated_ensemble import AbstractArbitratedEnsemble


class ArbitratedEnsembleAbstaining(AbstractArbitratedEnsemble):
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
    def __init__(self, meta_models, base_models, meta_error_metric='MSE', competence_threshold=10,
                 threshold_method='product', threshold_step=0.01, random_selection=None, meta_confidence_level=False):

        super().__init__(meta_models=meta_models, base_models=base_models, meta_error_metric=meta_error_metric)

        self.competence_threshold = competence_threshold
        self.initial_competence_threshold = competence_threshold
        self.threshold_method = threshold_method
        # TODO: better way to include new models
        self.threshold_step = threshold_step

        self.first_run = True
        self.previous_predictions = Queue()

        # TODO refactor creation of Normal Random Law for random selection

        self.random_selection = random_selection
        self.meta_confidence_level = meta_confidence_level
        self.outliers = []

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
            meta_models, base_models, meta_predictions, base_predictions = self._select_experts(np.array([X[i]]),
                                                                                                threshold=self.competence_threshold)

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

    def _select_experts(self, X, threshold=None):

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

        # TODO: Implement random variable on confidence level

        for i in range(len(self.meta_models)):
            z = self.meta_models[i]
            # TODO: handle negative errors predicted
            predicted_error = z.predict(X)[0]
            if self.meta_confidence_level:
                conf = z.get_confidence_score()
            else:
                conf = 1
            predicted_error = predicted_error * conf

            if np.abs(predicted_error) <= threshold:
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
        return 'ArbitratedEnsemble'

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
            # TODO : get previous predictions and compare to real value of y and see if we need to adapt threshold
            previous_predictions = self.previous_predictions.dequeue(r)
            # TODO: compare and see how to adapt threshold

            real_error = get_predictions_errors(y, previous_predictions, self.meta_error_metric)
            # TODO: change comparison value ==> initial tolerated error rather than dynamic one ?
            if real_error <= self.initial_competence_threshold:
                # We ara a facing a relatively true prediction
                # TODO: increase threshold to look for similarly competent classifiers
                self._update_threshold(method=self.threshold_method, change_step=self.threshold_step, increase=True)
            else:
                # We are facing a relatively false prediction
                # TODO: decrease threshold to prone less competent ones that were selected
                self._update_threshold(method=self.threshold_method, change_step=self.threshold_step, increase=False)

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

    def _update_threshold(self, method, change_step, increase):
        if not increase:
            step = -change_step
        else:
            step = change_step
        if method == 'static':
            pass
        elif method == 'sum':
            self.competence_threshold = self.competence_threshold + step
        elif method == 'product':
            self.competence_threshold = self.competence_threshold*(1 + step)
        else:
            raise ValueError('Unknown threshold update method', method)

    def _draw_random_values(self, distribution='normal', nb_items=1):

        if distribution == 'normal':
            return np.random.normal(1, 0.1, nb_items)
        elif distribution == 'binomial':
            return np.random.binomial(1, 0.95, nb_items)
        else:
            raise ValueError('Unknown ')







