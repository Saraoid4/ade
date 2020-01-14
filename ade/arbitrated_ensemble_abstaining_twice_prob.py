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

from ade.arbitrated_ensemble_abstaining import ArbitratedEnsembleAbstaining
from ade.selection_methods import select_experts_prob


class ArbitratedEnsembleAbstainingTwiceProb(ArbitratedEnsembleAbstaining):
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


    """

    # TODO: handle concept drift ad new models
    def __init__(self, meta_models, base_models, meta_error_metric='MAPE', meta_confidence_level=False,
                 output_file=None):

        super(ArbitratedEnsembleAbstainingTwiceProb, self).__init__(meta_models=meta_models, base_models=base_models,
                                                              meta_error_metric=meta_error_metric,
                                                              meta_confidence_level=meta_confidence_level,
                                                              output_file=output_file)
        if self.output_file:
            super(ArbitratedEnsembleAbstainingTwiceProb, self)._init_file()

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
            meta_models, base_models, meta_predictions, base_predictions, step_results = select_experts_prob(
                self.meta_models,
                self.base_models,
                np.array([X[i]]),
                self.meta_confidence_level)

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
                # TODO: get all base_predictions and all meta_predictions and all selected indices
                super(ArbitratedEnsembleAbstainingTwiceProb, self)._update_outputs(global_prediction=final_prediction,
                                                                              base_predictions=step_results[
                                                                                  'all_base_predictions'],
                                                                              meta_predictions=step_results[
                                                                                  'all_meta_predictions'],
                                                                              base_selected_idx=step_results[
                                                                                  'selected_experts_idx'])
            except Exception as exc:
                raise exc

        self.previous_predictions.enqueue(predictions)
        return predictions

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
        info = '{}:'.format(type(self).__name__)
        info += ' - ensemble_size: {}'.format(len(self.base_models))
        info += ' - meta_error_metric: {}'.format(self.meta_error_metric)
        return info

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
                real_p = 1 - np.abs(real_errors) / 100
                if real_p > 1 or real_p < 0:
                    random_error = np.random.binomial(1, sys.float_info.epsilon, 1)
                else:
                    random_error = np.random.binomial(1, real_p, 1)[0]
                if random_error:
                    m.partial_fit(X, y, weight)

    def get_model_name(self):
        model_name = super(ArbitratedEnsembleAbstainingTwiceProb, self).get_model_name()
        name_parts = [model_name, 'PROB']
        if self.meta_confidence_level:
            name_parts.append('CONF')
        name_parts.append('TWICE')
        return '_'.join(name_parts)

