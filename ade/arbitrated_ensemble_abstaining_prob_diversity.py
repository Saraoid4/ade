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

from diversity.diversity_factory import DiversityMeasuresFactory
from ade.arbitrated_ensemble_abstaining import ArbitratedEnsembleAbstaining
from ade.selection_methods import select_experts_prob_diversity, get_all_predictions, select_experts_prob


class ArbitratedEnsembleAbstainingProbDiversity(ArbitratedEnsembleAbstaining):
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
    def __init__(self, meta_models, base_models, meta_error_metric='MAPE', meta_confidence_level=False, diversity_method='sliding_window',
                 diversity_measure='correlation', diversity_threshold=0.7, n_sliding=200, fading_factor=None, output_file=None):


        if diversity_method == 'sliding_window':
            data = {'model_list': base_models, 'window_size': 200}
        elif diversity_method == 'fading factor':
            data = {'model_list': base_models, 'alpha': 0.995}
        elif diversity_method == 'incremental':
            data = {'model_list': base_models}
        else:
            raise NotImplementedError

        self.diversity_method = diversity_method
        # Diversity selection parameters
        self.diversity_measure = diversity_measure
        #TODO: Change threshold based on diversity measures
        self.correlation_threshold = diversity_threshold
        self.n_sliding = n_sliding
        self.fading_factor = fading_factor
        div = DiversityMeasuresFactory()
        self.diversity_evaluator = div.get_diversity_evaluator(diversity_method=diversity_method,
                                                               diversity_measure=diversity_measure, args_dict=data)

        super(ArbitratedEnsembleAbstainingProbDiversity, self).__init__(meta_models=meta_models,
                                                                        base_models=base_models,
                                                                        meta_error_metric=meta_error_metric,
                                                                        meta_confidence_level=meta_confidence_level,
                                                                        output_file=output_file)

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes=None, weight=None):
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
            if self.diversity_evaluator is not None:
                meta_models, base_models, meta_predictions, base_predictions, step_results = select_experts_prob_diversity(self.meta_models,
                                                                                                         self.base_models,
                                                                                                         np.array([X[i]]),
                                                                                                         self.diversity_evaluator,
                                                                                                         self.diversity_measure,
                                                                                                         self.meta_confidence_level,
                                                                                                         self.correlation_threshold,
                                                                                                         n_best=2)

            else:
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
        info = ''
        # Diversity selection parameters
        info += ' - diversity_measure: {}'.format(self.diversity_measure)
        info += ' - n_sliding: {}'.format(self.n_sliding) if self.n_sliding is not None else ''
        info += ' - fading_factor: {}'.format(self.fading_factor) if self.fading_factor is not None else ''
        info += ' - correlation_threshold: {}'.format(self.correlation_threshold)

        return info

    def get_model_name(self):
        model_name = super(ArbitratedEnsembleAbstainingProbDiversity, self).get_model_name()
        model_name += '_PROB'
        if self.meta_confidence_level:
            model_name += '_CONF'
        model_name = '_'.join([model_name, 'DIVERSITY', self.diversity_method, self.diversity_measure])
        return model_name

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

            if hasattr(self, 'diversity_evaluator'):
                if self.diversity_evaluator is not None:
                    if self.diversity_evaluator.get_type() == 'window_regression':
                        all_predictions = get_all_predictions(self.base_models, X)
                        self.diversity_evaluator.add(all_predictions)
                    elif self.diversity_evaluator.get_type() == 'window_classif':
                        all_predictions = get_all_predictions(self.base_models, X)
                        self.diversity_evaluator.add(all_predictions, list(range(len(self.base_models))), y)
                    elif self.diversity_evaluator.get_type() == 'ff_classif':
                        all_predictions = get_all_predictions(self.base_models, X)[0]
                        self.diversity_evaluator.add(all_predictions, list(range(len(self.base_models))), y)
                    else:
                        raise NotImplementedError

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

            if hasattr(self, 'diversity_evaluator'):
                if self.diversity_evaluator is not None:
                    if self.diversity_evaluator.get_type() == 'window_classif':
                        all_predictions = get_all_predictions(self.base_models, X)[0]
                        self.diversity_evaluator.add(all_predictions, list(range(len(self.base_models))), y)
                    if self.diversity_evaluator.get_type() == 'ff_classif':
                        all_predictions = get_all_predictions(self.base_models, X)[0]
                        self.diversity_evaluator.add(all_predictions, list(range(len(self.base_models))), y)


