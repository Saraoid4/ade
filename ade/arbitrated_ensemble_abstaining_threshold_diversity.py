#!/usr/bin/env python

__author__ = "Dihia BOULEGANE"
__copyright__ = ""
__credits__ = ["Dihia BOULEGANE"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Dihia BOULEGANE"
__email__ = "dihia.boulegane@telecom-paristech.fr"
__status__ = "Development"

from ade.arbitrated_ensemble_abstaining_threshold import ArbitratedEnsembleAbstainingThreshold
from diversity.diversity_factory import DiversityMeasuresFactory
from skmultiflow.utils import *
from utils.functions import *
from utils.metrics import *
from ade.selection_methods import select_experts_diversity


class ArbitratedEnsembleAbstainingThresholdDiversity(ArbitratedEnsembleAbstainingThreshold):

    def __init__(self, meta_models, base_models, meta_error_metric='MAPE', competence_threshold=30,
                 threshold_update_method='product', threshold_update_step=0.01, meta_confidence_level=False,
                 diversity_method='sliding_window', diversity_measure='disagree', diversity_threshold=0.7,
                 n_sliding=200, fading_factor=None, output_file=None):



        if diversity_method == 'sliding_window':
            data = {'model_list': base_models, 'window_size': n_sliding}
        elif diversity_method == 'fading factor':
            data = {'model_list': base_models, 'alpha': fading_factor}
        elif diversity_method == 'incremental':
            data = {'model_list': base_models}
        else:
            raise NotImplementedError

        self.diversity_method = diversity_method

        # Diversity selection parameters
        self.diversity_measure = diversity_measure
        #TODO: change threshold based on diversity_measure
        self.correlation_threshold = diversity_threshold
        self.n_sliding = n_sliding
        self.fading_factor = fading_factor
        div = DiversityMeasuresFactory()
        self.diversity_evaluator = div.get_diversity_evaluator(diversity_method=diversity_method, diversity_measure=diversity_measure, args_dict=data)

        super(ArbitratedEnsembleAbstainingThresholdDiversity, self).__init__(meta_models=meta_models,
                                                                             base_models=base_models,
                                                                             meta_error_metric=meta_error_metric,
                                                                             competence_threshold=competence_threshold,
                                                                             threshold_update_method=threshold_update_method,
                                                                             threshold_update_step=threshold_update_step,
                                                                             meta_confidence_level=meta_confidence_level,
                                                                             output_file=output_file)

        if self.output_file is not None:
            super(ArbitratedEnsembleAbstainingThresholdDiversity, self)._init_file()

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes=None, weight=None):

        """ performs a partial fit for all meta and base model with an update strategy of the threshold

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

        """

        # Add first predictions to buffer in the first run
        super(ArbitratedEnsembleAbstainingThresholdDiversity, self).partial_fit(X=X, y=y, classes=classes, weight=weight)

    def predict(self, X):

        """ Predicts target using the arbitrated ensemble model.

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
            meta_models, base_models, meta_predictions, base_predictions, step_results = select_experts_diversity(self.meta_models, self.base_models, np.array([X[i]]),
                                                                                                                  self.diversity_evaluator, self.diversity_measure,
                                                                                                                  self.meta_confidence_level,
                                                                                                                  self.competence_threshold, self.correlation_threshold,
                                                                                                                  2)

            if len(base_models) > 0:
                """ Sub set of experts found"""
                base_predictions = np.array(base_predictions).reshape(1, -1)
                weights = weight_predictions(meta_predictions)

            else:
                # No experts found ==> consider all base-mdoels
                meta_predictions = [z.predict([X[i]])[0] for z in self.meta_models]
                base_predictions = np.array([m.predict([X[i]])[0] for m in self.base_models]).reshape(1, -1)
                weights = weight_predictions(meta_predictions)

                # All base models have been selected
                step_results['selected_experts_idx'] = [1 for i in range(len(self.base_models))]

            try:
                final_prediction = get_aggregated_result(base_predictions, weights, method='weighted_average')
                predictions.append(final_prediction)

                # TODO: get all base_predictions and all meta_predictions and all selected indices
                super(ArbitratedEnsembleAbstainingThresholdDiversity, self)._update_outputs(
                    global_prediction=final_prediction,
                    base_predictions=step_results['all_base_predictions'],
                    meta_predictions=step_results['all_meta_predictions'],
                    base_selected_idx=step_results['selected_experts_idx']
                    )

            except Exception as exc:
                raise exc

        self.previous_predictions.enqueue(predictions)
        return predictions

    def predict_proba(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def get_class_type(self):
        return 'ArbitratedEnsembleAbstaining'

    def get_info(self):
        info = super(ArbitratedEnsembleAbstainingThresholdDiversity, self).get_info()
        # Diversity selection parameters
        info += ' - diversity_measure: {}'.format(self.diversity_measure)
        info += ' - n_sliding: {}'.format(self.n_sliding) if self.n_sliding is not None else ''
        info += ' - fading_factor: {}'.format(self.fading_factor) if self.fading_factor is not None else ''
        info += ' - correlation_threshold: {}'.format(self.correlation_threshold)

        return info
    
    def get_model_name(self):
        model_name = super(ArbitratedEnsembleAbstainingThresholdDiversity, self).get_model_name()

        model_name = '_'.join([model_name, 'DIVERSITY', self.diversity_method, self.diversity_measure])
        return model_name


