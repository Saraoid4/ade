
from abc import ABCMeta, abstractmethod
from skmultiflow.core.base import StreamModel
from skmultiflow.utils import *
from utils.metrics import *
from core.queue import Queue
from ade.selection_methods import get_all_predictions
from os.path import join
from ade.file_utils import init_file, update_file


class AbstractArbitratedEnsemble(StreamModel, metaclass=ABCMeta):
    """
    The abstract class for arbitrated ensemble models. The arbitrated architecture implemented in this class is designed
    to do Dynamic Ensemble Selection using meta-learning to predict future values of a stream.

    This approach is based on two levels of learning where each level learns on its own data and predicts its own values

    The base level learns to predict future values of the stream whereas each meta-learner is in charge to learn and
    predict future error of its base counter-part.

      Parameters
    ----------

    meta_models: Array-like  of  StreamModel
        Pool of StreamModel which stands for meta-learners where each model learns to predict the error rate of its base
        counter part

    base_models:
        Pool of StreamModel where each model learns to predict future values of the stream

    meta_error_metric: String
        Measure that will be used on the meta-level to estimate the error of the base-models. Could be MSE, MAE or MAPE
        or any other  measure to quantify models' error on a given test instance.

    """

    def __init__(self, meta_models, base_models, meta_error_metric='MAPE', output_file=None):
        super().__init__()
        self.meta_models = meta_models
        self.base_models = base_models

        self.first_run = True
        self.previous_predictions = Queue()

        self.meta_error_metric = meta_error_metric

        # Will be used to store global  and individual learners predictions and selected committee at each instance
        output_file = join(output_file, self.get_model_name() + '.csv')
        self.output_file = output_file

    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes=None, weight=None):
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

            # Information Diversity for MMR :
            if hasattr(self, 'symmetric_uncertainty'):
                all_predictions = get_all_predictions(self.base_models, X)
                if r > 1:
                    self.symmetric_uncertainty.update(all_predictions, y)
                else:
                    self.symmetric_uncertainty.update(all_predictions[0], y)


        else:
            r, _ = get_dimensions(X)
            # TODO: need to improve access to previous predictions
            # get previous predictions and compare to real value of y and see if we need to adapt threshold
            # previous_predictions = self.previous_predictions.dequeue(r)
            # compare and see how to adapt threshold
            # real_error = get_predictions_errors(y, previous_predictions, self.meta_error_metric)

            # Partial fit all base and meta learners
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
            #Information Diversity for MMR :
            if hasattr(self, 'symmetric_uncertainty'):
                all_predictions = get_all_predictions(self.base_models, X)
                if r > 1:
                    self.symmetric_uncertainty.update(all_predictions, y)
                else:
                    self.symmetric_uncertainty.update(all_predictions[0], y)

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def reset(self):
        self.base_models = []
        self.meta_models = []
        self.meta_error_metric = None

    def score(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def get_info(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_model_name(self):
        raise NotImplementedError

    def _init_file(self):
        base_models_info = [b_model.get_info() for b_model in self.base_models]
        init_file(self.output_file, len(self.base_models), base_models_info=base_models_info, global_info=self.get_info())


    def _update_outputs(self, global_prediction, base_predictions, meta_predictions, base_selected_idx):
        """ Update predictions and selection results"""
        update_file(self.output_file, global_prediction, base_predictions, meta_predictions, base_selected_idx)


