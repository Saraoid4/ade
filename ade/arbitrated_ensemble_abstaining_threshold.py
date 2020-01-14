
__author__ = "Dihia BOULEGANE"
__copyright__ = ""
__credits__ = ["Dihia BOULEGANE"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Dihia BOULEGANE"
__email__ = "dihia.boulegane@telecom-paristech.fr"
__status__ = "Development"


from ade.arbitrated_ensemble_abstaining import ArbitratedEnsembleAbstaining
from ade.selection_methods import get_all_predictions
from core.queue import Queue
from utils.metrics import *

from skmultiflow.utils import *


class ArbitratedEnsembleAbstainingThreshold(ArbitratedEnsembleAbstaining):

    def __init__(self, meta_models, base_models, meta_error_metric='MAPE', competence_threshold=20,
                 threshold_update_method='product', threshold_update_step=0.01, meta_confidence_level=False,
                 sequential_reweight=False, diversity_measure=None, output_file=None):

        # Threshold Abstaining parameters
        self.competence_threshold = competence_threshold
        self.initial_competence_threshold = competence_threshold
        self.threshold_update_method = threshold_update_method
        self.threshold_update_step = threshold_update_step

        self.sequential_reweight = sequential_reweight
        self.seq_diversity_measure = diversity_measure

        self.first_run = True
        self.previous_predictions = Queue()
        self.meta_confidence_level = meta_confidence_level

        super(ArbitratedEnsembleAbstainingThreshold, self).__init__(meta_models=meta_models, base_models=base_models,
                                                                    meta_error_metric=meta_error_metric,
                                                                    output_file=output_file)

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
        r, _ = get_dimensions(X)
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
                # Add first predictions to buffer in the first run
                # if self.diversity_evaluator is not None:
            if hasattr(self, 'diversity_evaluator'):
                if self.diversity_evaluator is not None:
                    all_predictions = get_all_predictions(self.base_models, X)
                    if r == 1:
                        all_predictions = all_predictions[0]
                    if self.diversity_evaluator.get_type() == 'window_regression':
                        if self.diversity_evaluator.get_class_type() =='window_dissimilarity':
                            self.diversity_evaluator.update_(all_predictions, y)
                        elif self.diversity_evaluator.get_class_type() == 'window_redundancy':
                            self.diversity_evaluator.update_(all_predictions, y)
                        elif self.diversity_evaluator.get_class_type() == 'window_correlation':
                            self.diversity_evaluator.add(all_predictions)
                        else:
                            raise NotImplementedError
                    elif self.diversity_evaluator.get_type() == 'window_classif':
                        self.diversity_evaluator.add(all_predictions, list(range(len(self.base_models))), y)
                    elif self.diversity_evaluator.get_type() == 'ff_classif':
                        self.diversity_evaluator.add(all_predictions[0], list(range(len(self.base_models))), y)
                    else:
                        raise NotImplementedError
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
                m.partial_fit(X, y, weight)
            if hasattr(self, 'diversity_evaluator'):
                if self.diversity_evaluator is not None:
                    all_predictions = get_all_predictions(self.base_models, X)
                    if r == 1:
                        all_predictions = all_predictions[0]
                    if self.diversity_evaluator.get_type() == 'window_classif':
                        self.diversity_evaluator.add(all_predictions, list(range(len(self.base_models))), y)
                    if self.diversity_evaluator.get_type() == 'ff_classif':
                        self.diversity_evaluator.add(all_predictions, list(range(len(self.base_models))), y)
                    if self.diversity_evaluator.get_type() == 'window_regression':
                        if self.diversity_evaluator.get_class_type() == 'window_dissimilarity':
                            self.diversity_evaluator.update_(all_predictions, y)
                        elif self.diversity_evaluator.get_class_type() == 'window_redundancy':
                            self.diversity_evaluator.update_(all_predictions, y)
                        elif self.diversity_evaluator.get_class_type() == 'window_correlation':
                            self.diversity_evaluator.add(all_predictions)

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
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def get_class_type(self):
        return 'ArbitratedEnsembleAbstainingThreshold'

    def get_info(self):
        info = super(ArbitratedEnsembleAbstainingThreshold, self).get_info()
        info += ' - competence_threshold: {}'.format(self.competence_threshold)
        info += ' - threshold_update_method: {}'.format(self.threshold_update_method)
        info += ' - threshold_update_step: {}'.format(self.threshold_update_step)

        return info

    def _update_file(self, global_prediction, base_predictions, meta_predictions, base_selected_idx):
        super(ArbitratedEnsembleAbstainingThreshold, self)._update_file(global_prediction, base_predictions, meta_predictions, base_selected_idx)

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
            self.competence_threshold = self.competence_threshold * (1 + step)
        else:
            raise ValueError('Unknown threshold update method', method)

    def get_model_name(self):
        model_name = super(ArbitratedEnsembleAbstainingThreshold, self).get_model_name()
        # TODO: change to format
        model_name += '_THRESHOLD_' + str(self.competence_threshold)+ '_'
        # Check parameters
        model_name += self.threshold_update_method.upper()
        if self.sequential_reweight:
            model_name = '_'.join([model_name, 'SEQ_W', self.seq_diversity_measure.upper()])
        return model_name



