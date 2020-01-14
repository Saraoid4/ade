from abc import ABCMeta, abstractmethod
from skmultiflow.utils import FastComplexBuffer, FastBuffer
import numpy as np


class WindowMeasures(metaclass=ABCMeta):

    def __init__(self, model_list=None, window_size=200):
        if model_list is not None:
            self.n_models = len(model_list)
        else:
            self.n_models = 0
        self.window_size = window_size
        self.last_components_predictions = FastComplexBuffer(window_size, self.n_models)
        self.show_plot = False

    def reset(self):
        self.n_models = 0
        self.last_components_predictions = FastComplexBuffer(self.window_size, self.n_models)
        self.show_plot = False

    def update(self, predictions):
        if len(predictions) == 1:
            predictions = predictions[0]
        self.last_components_predictions.add_element(predictions)

    @abstractmethod
    def get_info(self):
        raise NotImplementedError

    @abstractmethod
    def get_class_type(self):
        raise NotImplementedError

    @staticmethod
    def get_type():
        return 'window_regression'


class WindowClassificationMeasures(WindowMeasures):
    def __init__(self, model_list, window_size=200):
        super().__init__(model_list=model_list, window_size=window_size)
        self.last_true_value = FastBuffer(window_size)
        self.binary_values = [[] for _ in range(self.n_models)]
        self.estimator = np.zeros((self.n_models, self.n_models))

    def reset(self):
        super().reset()

    def update(self, predictions):
        super().update(predictions=predictions)

    def binarize(self, y_true, predictions):
        std = np.std(predictions)
        for i in sorted(range(self.n_models)):
            if y_true - std <= predictions[i] <= y_true + std:
                self.binary_values[i].append(1)
            else:
                self.binary_values[i].append(0)
        return self.binary_values

    def add(self, predictions, model_list=None, y=None):
        raise NotImplementedError

    def get_pairwise_diversity(self, i, j):
        raise NotImplementedError

    def get_diversity_matrix(self, estimation_method=None):
        raise NotImplementedError

    def get_experts_diversity(self, model_list=None, sub_models=None):
        raise NotImplementedError

    @staticmethod
    def diversity_contingency(i, j, expectedi, expectedj):
        value = 0
        if i == expectedi and j == expectedj:
            value += 1
        return value
    @staticmethod
    def get_type():
        return 'window_classif'

    @abstractmethod
    def get_info(self):
        raise NotImplementedError

    @abstractmethod
    def get_class_type(self):
        raise NotImplementedError


class FadingFactorMeasures(metaclass=ABCMeta):

    def __init__(self, model_list, alpha):
        if model_list is not None:
            self.n_models = len(model_list)
        else:
            self.n_models = 0
        self.last_predicted_values = FastBuffer(self.n_models)
        self.binary_values = [0 for i in range(self.n_models)]
        self.alpha = alpha
        self.std = 0.0
        self.estimator = np.zeros((self.n_models, self.n_models))
        self.increment = 0.0
        self.last_true_value = FastBuffer(1)

    def binarize(self, y_true, predictions):
        std = np.std(predictions)
        for i in range(self.n_models):
            if y_true - std <= predictions[i] <= y_true + std:
                self.binary_values[i] = 1
            else:
                self.binary_values[i] = 0
        return self.binary_values

    @staticmethod
    def diversity_contingency(i, j, expectedi, expectedj):
        value = 0
        if i == expectedi and j == expectedj:
            value += 1
        return value

    @abstractmethod
    def add(self, predictions, model_list=None, y_true=None):
        raise NotImplementedError

    @abstractmethod
    def get_pairwise_diversity(self, i, j):
        raise NotImplementedError

    @abstractmethod
    def get_diversity_matrix(self):
        raise NotImplementedError

    @abstractmethod
    def get_info(self):
        raise NotImplementedError

    @abstractmethod
    def get_class_type(self):
        raise NotImplementedError\

    @staticmethod
    def get_type():
        return 'ff_classif'
