from abc import ABC

from skmultiflow.utils import get_dimensions

from ade.arbitrated_ensemble_abstaining import ArbitratedEnsembleAbstaining
from ade.selection_methods import get_experts_prediction_abstaining


class ArbitratedEnsembleAbstainingBest(ArbitratedEnsembleAbstaining, ABC):
    """ ArbitratedEnsembleAbstainingBest

    The arbitrated architecture implemented in this class is designed to do Dynamic Ensemble Selection using $
    meta-learning to predict future values of a stream. It's based on an abstaining policy to select  n_best most
    confident base-models only to contribute to the final output and weight them inversely to the estimated error by the
    the meta-models predicted errors.

    Parameters
    ----------
    meta_models: Array-like  of  StreamModel
        Pool of StreamModel where each model learns to predict the error rate of its base counter part
    base_models: Array-like  of  StreamModel
        Pool of StreamModel where each model learns to predict values of the stream
    n_best: float
        Percentage of base-models to mbe selected from the pool
    """

    def __init__(self, meta_models, base_models, meta_error_metric='MAPE', output_file=None, n_best=1):

        # TODO: check compared to length of models
        self.n_best = n_best
        
        super(ArbitratedEnsembleAbstainingBest, self).__init__(meta_models=meta_models, base_models=base_models,
                                                               meta_error_metric=meta_error_metric,
                                                               output_file=output_file)
        if self.output_file:
            super(ArbitratedEnsembleAbstainingBest, self)._init_file()



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
        super(ArbitratedEnsembleAbstainingBest, self).partial_fit(X, y, classes,weight)

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
            final_prediction, step_results = get_experts_prediction_abstaining(X = X,
                                                                               meta_models=self.meta_models,
                                                                               base_models=self.base_models,
                                                                               selection_method='n_best',
                                                                               sequential_reweight=None,
                                                                               competence_threshold=None,
                                                                               selection_ratio=None,
                                                                               n_best=self.n_best,
                                                                               diversity_evaluator=None)

            predictions.append(final_prediction)
            super(ArbitratedEnsembleAbstainingBest, self)._update_outputs(
                global_prediction=final_prediction,
                base_predictions=step_results['all_base_predictions'],
                meta_predictions=step_results['all_meta_predictions'],
                base_selected_idx=step_results['selected_experts_idx'])
        self.previous_predictions.enqueue(predictions)
        return predictions
    
    def get_model_name(self):
        model_name = super(ArbitratedEnsembleAbstainingBest, self).get_model_name()
        model_name = '_'.join([model_name, 'BEST', str(self.n_best)])
        return model_name





