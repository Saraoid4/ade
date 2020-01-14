import numpy as np
import pandas as pd

from abc import ABCMeta 
from skmultiflow.utils import FastComplexBuffer, FastBuffer
from skmultiflow.core.base import StreamModel


from utils.functions import compute_agreement
from utils.metrics import get_predictions_errors
from ade.arbitrated_ensemble_abstaining import ArbitratedEnsembleAbstaining
from ade.selection_methods import get_experts_prediction_abstaining


class ADEKrippendorff(ArbitratedEnsembleAbstaining, metaclass=ABCMeta):
    """ Ensemble method using krippendorff alpha measure for disagreement to spot concept drift
    Parameters
    ----------
    base_models : list, array-like
        list of learning models
    disagreement_ath : float
        value of krippendorff alpha value for threshold to assess disagreement
    """
    def __init__(self, meta_models, base_models, window_size, disagreement_th, selection_method='probability', competence_th=None, output_file=None):
        super(ADEKrippendorff, self).__init__(meta_models=meta_models, base_models=base_models, meta_error_metric='MAPE', output_file=output_file)

        self.disagreement_th = disagreement_th
        self.previous_predictions = FastComplexBuffer(max_size=window_size, width=len(self.base_models))
        self.previous_true_values = FastBuffer(max_size=window_size)
        self.window_size = window_size
        # self.select_experts = select_experts_prob if selection_method == 'prob' else select_experts_threshold
        # TODO: to be deleted in the future
        self.stored_computed_agreement = []
        self.stored_correlations = []
        self.selection_method = selection_method

        if self.output_file:
            super(ADEKrippendorff, self)._init_file()

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes=None, weight=None):
        super(ADEKrippendorff, self).partial_fit(X, y, weight)
        # If first partial fit do not store true label, if not store true label y 
        if self.previous_predictions.current_size > 0:
            # Store True Value
            self.previous_true_values.add_element(y)

    def predict(self, X):
        final_prediction, step_results = get_experts_prediction_abstaining(X=X, meta_models=self.meta_models,
                                                                           base_models=self.base_models,
                                                                           selection_method=self.selection_method,
                                                                           sequential_reweight=False,
                                                                           diversity_measure = None,
                                                                           competence_threshold=None,
                                                                           diversity_evaluator=None)
        
       
        
        # Store previous base predictionss
        base_predictions_vector = [None] * len(self.base_models)
        for i in step_results['selected_experts_idx']:
            base_predictions_vector[i] = step_results['all_base_predictions'][i]

        self.previous_predictions.add_element(base_predictions_vector)

        # Building reliability matrix 
        reliability_matrix = pd.DataFrame()
        # Compute and print disagreement
        # TODO: do it incremental for the data frame and disagrement computation
        for i in range(self.previous_predictions.current_size):
            reliability_matrix[i] = self.previous_predictions.buffer[i]

        inner_agreement = compute_agreement(reliability_matrix)
        

        self.stored_computed_agreement.append(inner_agreement)
        
       
        super(ADEKrippendorff, self)._update_outputs(global_prediction=final_prediction,
                                                     base_predictions=step_results['all_base_predictions'],
                                                     meta_predictions=step_results['all_meta_predictions'],
                                                     base_selected_idx=step_results['selected_experts_idx'])
        
        return [final_prediction]

    def predict_proba(self, X):
        raise NotImplementedError  
    def reset(self):
        raise NotImplementedError
    def score(self, X, y):
        raise NotImplementedError

    def get_class_type(self):
        return 'ADE Krippendorff'

    def get_info(self):
        info = '{}:'.format(type(self).__name__)
        info += ' - ensemble_size: {}'.format(len(self.base_models))
        info += ' - window_size:{}'.format(self.window_size)
        return info
    
    def get_model_name(self):
        model_name = super(ADEKrippendorff, self).get_model_name()
        model_name = '_'.join([model_name, 'KRIPP'])

        return model_name
    