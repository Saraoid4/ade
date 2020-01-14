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


class ArbitratedEnsembleAbstainingProb(ArbitratedEnsembleAbstaining):
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
    def __init__(self, meta_models, base_models, meta_error_metric='MAPE', meta_confidence_level=True,
                 sequential_reweight=False, diversity_measure='correlation', diversity_method='sliding_window',n_sliding=200, output_file=None):

        self.sequential_reweight = sequential_reweight
        self.seq_diversity_measure = diversity_measure
        super().__init__(meta_models=meta_models, base_models=base_models, meta_error_metric=meta_error_metric,
                         meta_confidence_level=meta_confidence_level, output_file=output_file)



        if self.sequential_reweight:
            # TODO : Update and change hard coding
            self.n_sliding = n_sliding
            # Will be used with correlation/redundancy/dissimilarity and sliding window
            data = {'model_list': self.base_models, 'window_size': self.n_sliding}
            # Diversity selection parameters
            self.diversity_method = diversity_method
            div = DiversityMeasuresFactory()

            self.diversity_evaluator = div.get_diversity_evaluator(diversity_method=self.diversity_method,
                                                                   diversity_measure=self.seq_diversity_measure,
                                                                   args_dict=data)
        if self.output_file:
            super(ArbitratedEnsembleAbstainingProb, self)._init_file()

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
        super(ArbitratedEnsembleAbstainingProb, self).partial_fit(X, y, weight)

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
                                                                                   selection_method='probability',
                                                                                   sequential_reweight=self.sequential_reweight,
                                                                                   diversity_measure=self.seq_diversity_measure,
                                                                                   competence_threshold=None,
                                                                                   diversity_evaluator=self.diversity_evaluator)
            else:
                final_prediction, step_results = get_experts_prediction_abstaining(X=X, meta_models=self.meta_models,
                                                                                   base_models=self.base_models,
                                                                                   selection_method='probability',
                                                                                   sequential_reweight=self.sequential_reweight,
                                                                                   diversity_measure =None,
                                                                                   competence_threshold=None,
                                                                                   diversity_evaluator=None)

            predictions.append(final_prediction)
            # TODO: get all base_predictions and all meta_predictions and all selected indices
            super(ArbitratedEnsembleAbstainingProb, self)._update_outputs(global_prediction=final_prediction,
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
        self.base_models = []

    def get_class_type(self):
        return 'ArbitratedEnsembleAbstainProb'

    def get_info(self):
        info = super(ArbitratedEnsembleAbstainingProb, self).get_info()
        info += ' - sequential_reweight: {}'.format(self.sequential_reweight)
        info += ' - seq-diversity-measure: {}'.format(self.seq_diversity_measure)
        return info

    def get_model_name(self):
        model_name = super(ArbitratedEnsembleAbstainingProb, self).get_model_name()
        model_name = '_'.join([model_name, 'PROB'])

        if self.meta_confidence_level:
            model_name += '_CONF'
        if self.sequential_reweight:
            model_name = '_'.join([model_name, 'SEQ_W', self.seq_diversity_measure.upper()])
        return model_name







