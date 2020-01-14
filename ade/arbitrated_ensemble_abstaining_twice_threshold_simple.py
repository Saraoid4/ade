from skmultiflow.drift_detection import PageHinkley

__author__ = "Dihia BOULEGANE"
__copyright__ = ""
__credits__ = ["Dihia BOULEGANE"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Dihia BOULEGANE"
__email__ = "dihia.boulegane@telecom-paristech.fr"
__status__ = "Development"


from ade.arbitrated_ensemble_abstaining_threshold import ArbitratedEnsembleAbstainingThreshold
from utils.functions import *
from utils.metrics import *
from skmultiflow.utils import *
from ade.selection_methods import select_experts_threshold


class ArbitratedEnsembleAbstainingTwiceThresholdSimple(ArbitratedEnsembleAbstainingThreshold):

    def __init__(self, meta_models, base_models, meta_error_metric='MAPE', competence_threshold=10,
                 threshold_update_method='product', threshold_update_step=0.01, meta_confidence_level=False,
                 output_file=None):

        super(ArbitratedEnsembleAbstainingTwiceThresholdSimple, self).__init__(meta_models=meta_models,
                                                                               base_models=base_models,
                                                                               meta_error_metric=meta_error_metric,
                                                                               competence_threshold=competence_threshold,
                                                                               threshold_update_method=threshold_update_method,
                                                                               threshold_update_step=threshold_update_step,
                                                                               meta_confidence_level=meta_confidence_level,
                                                                               output_file=output_file)
        if self.output_file:
            super(ArbitratedEnsembleAbstainingTwiceThresholdSimple, self)._init_file()

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
        real_error_all = {}
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

            # get previous predictions and compare to real value of y and see if we need to adapt threshold
            previous_predictions = self.previous_predictions.dequeue(r)
            # compare and see how to adapt threshold
            real_error = get_predictions_errors(y, previous_predictions, self.meta_error_metric)
            # compare error to  initial tolerated error rather than dynamic one ?
            if real_error <= self.initial_competence_threshold:
                # We ara a facing a relatively true prediction
                # increase threshold to look for similarly competent classifiers
                self._update_threshold(method=self.threshold_update_method, change_step=self.threshold_update_step,
                                       increase=True)
            else:
                # We are facing a relatively false prediction
                # decrease threshold to prone less competent ones that were selected
                self._update_threshold(method=self.threshold_update_method, change_step=self.threshold_update_step,
                                       increase=False)
            for z, m in zip(self.meta_models, self.base_models):
                # Incremental training of base base models
                # Getting base models predictions
                predictions = m.predict(X)
                # Computing real incurred errors
                real_errors = np.array([get_predictions_errors(real_y, estimated_y, self.meta_error_metric)
                                        for real_y, estimated_y in zip(y, predictions)])
                # Incremental learning of meta-models
                z.partial_fit(X, real_errors, weight)
                if real_errors[0] <= self.competence_threshold:
                    m.partial_fit(X, y)



    def _get_real_experts(self, real_errors):
        """

        Parameters
        ----------
        real_errors = dict{m:real_error}

        Returns
        -------
        List of base-models corresponding to real experts according to competence_threshold
        """
        base_experts = [m for m in real_errors.keys() if real_errors[m] <= self.competence_threshold]

        return base_experts

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
        # TODO: more refactoring on predict code
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            meta_models, base_models, meta_predictions, base_predictions, step_results = select_experts_threshold(
                self.meta_models,
                self.base_models,
                np.array([X[i]]),
                self.competence_threshold)

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
                super(ArbitratedEnsembleAbstainingTwiceThresholdSimple, self)._update_outputs(
                    global_prediction=final_prediction,
                    base_predictions=step_results['all_base_predictions'],
                    meta_predictions=step_results['all_meta_predictions'],
                    base_selected_idx=step_results['selected_experts_idx'])
            except Exception as exc:
                raise exc

        self.previous_predictions.enqueue(predictions)
        return predictions

    def predict_proba(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def get_class_type(self):
        return 'ArbitratedEnsembleAbstainingTwiceThreshold'

    def get_info(self):
        info = super(ArbitratedEnsembleAbstainingTwiceThresholdSimple, self).get_info()
        return info

    def get_model_name(self):
        model_name = super(ArbitratedEnsembleAbstainingTwiceThresholdSimple, self).get_model_name()
        return '_'.join([model_name, 'TWICE'])





