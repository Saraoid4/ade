__author__ = "Dihia BOULEGANE"
__copyright__ = ""
__credits__ = ["Dihia BOULEGANE"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Dihia BOULEGANE"
__email__ = "dihia.boulegane@telecom-paristech.fr"
__status__ = "Development"


from skmultiflow.utils import *
from utils.functions import *
from utils.metrics import *
from ade.abstract_arbitrated_ensemble import AbstractArbitratedEnsemble
from os.path import join


class ArbitratedEnsemble(AbstractArbitratedEnsemble):

    """ ArbitratedEnsemble

    This is the simplest version of arbitrated ensemble approach where all base-learners contribute to the final
    prediction on a given instance of the stream. When a test instance X_i of the stream comes, each base model M^j
    outputs the estimated  value y^j_i. Each meta-model Z^j predicts an estimated error e^j_i of its base
    counter-part. The final output is a weighted average of all outputs y^j_i using weights w^j_i inversely related
    to the predicted e^j_i.


    """
    # TODO: add parameters docstring

    def __init__(self, meta_models, base_models, meta_error_metric='MAPE', output_file=None):
        super(ArbitratedEnsemble, self).__init__(meta_models=meta_models, base_models=base_models,
                                                 meta_error_metric=meta_error_metric, output_file=output_file)

    def fit(self, X, y, classes=None, weight=None):
        """ Fit the model under the batch setting (Not implemented in this case).

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

        Returns
        -------
        self
        """
        raise NotImplementedError

    def partial_fit(self, X, y, classes=None, weight=None):
        super(ArbitratedEnsemble, self).partial_fit(X, y, classes,weight)

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
        r, _ = get_dimensions(X)
        predictions = []

        for i in range(r):
            # TODO: USe selection_methods.get_all_prediction_data_frame
            meta_predictions = [z.predict([X[i]])[0] for z in self.meta_models]
            weights = weight_predictions(meta_predictions)
            base_predictions = np.array([m.predict([X[i]])[0] for m in self.base_models]).reshape(1, -1)
            final_prediction = get_aggregated_result(base_predictions, weights, method='weighted_average')
            predictions.append(final_prediction)

        return predictions

    def get_class_type(self):
        return 'ArbitratedEnsemble'

    def score(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def get_info(self):
        info = '{}:'.format(type(self).__name__)
        info += ' - ensemble_size: {}'.format(len(self.base_models))
        info += ' - meta_error_metric: {}'.format(self.meta_error_metric)
        return info

    def get_model_name(self):
        return 'ADE_WEIGHTED_' + self.meta_error_metric + '_SIZE_' + str(len(self.base_models))

