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
from skmultiflow.utils import *
from ade.selection_methods import get_experts_prediction_abstaining
from diversity.diversity_factory import DiversityMeasuresFactory


class ArbitratedEnsembleAbstainingThresholdSimple(ArbitratedEnsembleAbstainingThreshold):

    def __init__(self, meta_models, base_models, meta_error_metric='MAPE', competence_threshold=30,
                 threshold_update_method='product', threshold_update_step=0.01, meta_confidence_level=False,
                 sequential_reweight=False, diversity_measure='correlation', output_file=None):


        super(ArbitratedEnsembleAbstainingThresholdSimple, self).__init__(meta_models=meta_models,
                                                                          base_models=base_models,
                                                                          meta_error_metric=meta_error_metric,
                                                                          competence_threshold=competence_threshold,
                                                                          threshold_update_method=threshold_update_method,
                                                                          threshold_update_step=threshold_update_step,
                                                                          meta_confidence_level=meta_confidence_level,
                                                                          sequential_reweight=sequential_reweight,
                                                                          diversity_measure=diversity_measure,
                                                                          output_file=output_file)

        if self.sequential_reweight:
            self.n_sliding = 200
            # Will be used only with correlation and sliding window
            data = {'model_list': self.base_models, 'window_size': self.n_sliding}
            # Diversity selection parameters

            self.diversity_method = 'sliding_window'
            div = DiversityMeasuresFactory()

            self.diversity_evaluator = div.get_diversity_evaluator(diversity_method=self.diversity_method,
                                                                   diversity_measure=self.seq_diversity_measure, args_dict=data)

        if self.output_file:
            super(ArbitratedEnsembleAbstainingThresholdSimple, self)._init_file()

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

        super(ArbitratedEnsembleAbstainingThresholdSimple, self).partial_fit(X, y, classes,weight)

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
        predictions = []
        r, _ = get_dimensions(X)

        for i in range(r):
            if self.sequential_reweight:
                final_prediction, step_results = get_experts_prediction_abstaining(X=X, meta_models=self.meta_models,
                                                                     base_models=self.base_models,
                                                                     selection_method='threshold',
                                                                     sequential_reweight=self.sequential_reweight,
                                                                     diversity_measure= self.seq_diversity_measure,
                                                                     competence_threshold=self.competence_threshold,
                                                                     diversity_evaluator=self.diversity_evaluator)
            else:
                final_prediction, step_results = get_experts_prediction_abstaining(X=X, meta_models=self.meta_models,
                                                                                   base_models=self.base_models,
                                                                                   selection_method='threshold',
                                                                                   sequential_reweight=self.sequential_reweight,
                                                                                   diversity_measure=None,
                                                                                   competence_threshold=self.competence_threshold,
                                                                                   diversity_evaluator=None)

            predictions.append(final_prediction)

            super(ArbitratedEnsembleAbstainingThresholdSimple, self)._update_outputs(
                global_prediction=final_prediction,
                base_predictions=step_results['all_base_predictions'],
                meta_predictions=step_results['all_meta_predictions'],
                base_selected_idx=step_results['selected_experts_idx'])

        self.previous_predictions.enqueue(predictions)
        return predictions

    def predict_proba(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def get_class_type(self):
        return 'ArbitratedEnsembleAbstainingThresholdSimple'

    def get_info(self):
        info = super(ArbitratedEnsembleAbstainingThresholdSimple, self).get_info()
        info += ' - sequential_reweight: {}'.format(self.sequential_reweight)
        return info

    def get_model_name(self):
        model_name = super(ArbitratedEnsembleAbstainingThresholdSimple, self).get_model_name()

        return model_name










