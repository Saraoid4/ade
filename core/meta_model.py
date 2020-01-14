import sys
import numpy as np
from skmultiflow.core.base import StreamModel
from skmultiflow.trees import RegressionHoeffdingTree, RegressionHAT
from skmultiflow.utils import *
from .queue import Queue


class MetaModel(StreamModel):

    def __init__(self, model_type='HATREE', confidence_fading_factor=0.995):
        super().__init__()
        if model_type == 'HATREE':
            self.model = RegressionHAT()
        elif model_type == 'HTREE':
            self.model = RegressionHoeffdingTree()
        # TODO : add random init and change 0

        self.confidence_level = 0
        self.previous_predictions = Queue()

        self.confidence_fading_factor = confidence_fading_factor
        self.examples_seen = 0
        self.total_square_error = 0
        self.total_absolute_error = 0

    def fit(self, X, y, classes=None, weight=None):
        self.model.fit(X, y, classes=classes, weight=weight)

    def partial_fit(self, X, y, weight=None):
        self.model.partial_fit(X, y, weight=weight)
        # TODO: update confidence level with the new label

        # Compute real error
        r, _ = get_dimensions(X)
        self.examples_seen += r
        if self.previous_predictions.has_next():
            estimated_y = self.previous_predictions.dequeue(r)
        else:
            estimated_y = self.model.predict(X)

        # update total square and absolute error
        self.total_absolute_error = np.abs(estimated_y - y).sum() + self.confidence_fading_factor * self.total_absolute_error
        self.total_square_error = np.square(estimated_y - y).sum() + self.confidence_fading_factor * self.total_square_error

        self.confidence_level = np.exp(-self.total_square_error / self.examples_seen)
        # Avoiding zero confidence to avoid all wrights to be set to zero
        if self.confidence_level == 0:
            self.confidence_level = sys.float_info.epsilon

    def predict(self, X):
        meta_error = self.model.predict(X)
        self.previous_predictions.enqueue(meta_error)
        return meta_error

    def reset(self):
        self.model = None
        self.confidence_level = 0

    def predict_proba(self, X):
        raise NotImplementedError()

    def get_confidence_score(self):
        # This will be used to return confidence level as exp(-mse)
        return self.confidence_level

    def get_info(self):
        return self.model.get_info()

    def score(self, X, y):
        raise NotImplementedError()


