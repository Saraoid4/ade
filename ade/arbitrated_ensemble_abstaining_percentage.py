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

from ade.arbitrated_ensemble_abstaining import ArbitratedEnsembleAbstaining
from ade.selection_methods import get_experts_prediction_abstaining
from diversity.diversity_factory import DiversityMeasuresFactory


class ArbitratedEnsembleAbstainingPercentage(ArbitratedEnsembleAbstaining):
    """ ArbitratedEnsembleAbstainingPercentage

    The arbitrated architecture implemented in this class is designed to do Dynamic Ensemble Selection using $
    meta-learning to predict future values of a stream. It's based on an abstaining policy to select  alpha % most
    confident base-models only to contribute to the final output and weight them inversely to the estimated error by the
    the meta-models predicted errors.

    Parameters
    ----------
    meta_models: Array-like  of  StreamModel
        Pool of StreamModel where each model learns to predict the error rate of its base counter part
    base_models: Array-like  of  StreamModel
        Pool of StreamModel where each model learns to predict values of the stream
    selection_ratio: float
        Percentage of base-models to mbe selected from the pool
    """

    def __init__(self, meta_models, base_models, meta_error_metric='MAPE', meta_confidence_level=True, output_file=None,
                 selection_ratio=0.5, sequential_reweight=False, diversity_measure='correlation', diversity_method='sliding_window'):


        self.selection_ratio = selection_ratio

        self.sequential_reweight = sequential_reweight
        if self.sequential_reweight:
            self.n_sliding = 200
            # Will be used only with correlation and sliding window
            data = {'model_list': base_models, 'window_size': self.n_sliding}

            # Diversity selection parameters
            self.diversity_measure = diversity_measure
            self.diversity_method = diversity_method
            
            div = DiversityMeasuresFactory()

            self.diversity_evaluator = div.get_diversity_evaluator(diversity_method=self.diversity_method,
                                                                   diversity_measure=self.diversity_measure,
                                                                   args_dict=data)

        super().__init__(meta_models=meta_models, base_models=base_models, meta_error_metric=meta_error_metric,
                         output_file=output_file)

        if self.output_file:
            super(ArbitratedEnsembleAbstainingPercentage, self)._init_file()

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes=None,weight=None):
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
        super(ArbitratedEnsembleAbstainingPercentage, self).partial_fit(X, y, classes,weight)

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

            if self.sequential_reweight:
                final_prediction, step_results = get_experts_prediction_abstaining(X=X, meta_models=self.meta_models,
                                                                                   base_models=self.base_models,
                                                                                   selection_method='percentage',
                                                                                   sequential_reweight=self.sequential_reweight,
                                                                                   competence_threshold=None,
                                                                                   selection_ratio=self.selection_ratio,
                                                                                   diversity_evaluator=self.diversity_evaluator)
            else:
                final_prediction, step_results = get_experts_prediction_abstaining(X=X, meta_models=self.meta_models,
                                                                                   base_models=self.base_models,
                                                                                   selection_method='percentage',
                                                                                   sequential_reweight=self.sequential_reweight,
                                                                                   competence_threshold=None,
                                                                                   selection_ratio=self.selection_ratio,
                                                                                   diversity_evaluator=None)
            predictions.append(final_prediction)
            # TODO: get all base_predictions and all meta_predictions and all selected indices
            super(ArbitratedEnsembleAbstainingPercentage, self)._update_outputs(
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

    def reset(self):
        self.meta_models = []
        # super(ArbitratedEnsembleAbstainingPercentage, self).reset()

    def get_class_type(self):
        return 'ArbitratedEnsembleAbstainPercentage'

    def get_info(self):
        info = super(ArbitratedEnsembleAbstainingPercentage, self).get_info()
        info += ' - selection_ratio: {}'.format(self.selection_ratio)
        return info

    def get_model_name(self):
        model_name = super(ArbitratedEnsembleAbstainingPercentage, self).get_model_name()
        model_name += '_PERCENT'
        model_name += '_'+ str(self.selection_ratio)
        if self.sequential_reweight:
            model_name += '_SEQ_W'
        return model_name
