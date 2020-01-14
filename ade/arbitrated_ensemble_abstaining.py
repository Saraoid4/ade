#!/usr/bin/env python


__author__ = "Dihia BOULEGANE"
__copyright__ = ""
__credits__ = ["Dihia BOULEGANE"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Dihia BOULEGANE"
__email__ = "dihia.boulegane@telecom-paristech.fr"
__status__ = "Development"

from ade.abstract_arbitrated_ensemble import AbstractArbitratedEnsemble


class ArbitratedEnsembleAbstaining(AbstractArbitratedEnsemble):

    def __init__(self, meta_models, base_models, meta_error_metric='MAPE', meta_confidence_level=False, output_file=None):

        self.meta_confidence_level = meta_confidence_level

        super(ArbitratedEnsembleAbstaining, self).__init__(meta_models=meta_models,
                                                           base_models=base_models,
                                                           meta_error_metric=meta_error_metric,
                                                           output_file=output_file)


    def fit(self, X, y, classes=None, weight=None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes=None, weight=None):
        super(ArbitratedEnsembleAbstaining, self).partial_fit(X, y, classes, weight)

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
        raise NotImplementedError()

    def predict_proba(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def get_class_type(self):
        return 'ArbitratedEnsembleAbstaining'

    def get_info(self):
        info = '{}:'.format(type(self).__name__)
        info += ' - ensemble_size: {}'.format(len(self.base_models))
        info += ' - meta_error_metric: {}'.format(self.meta_error_metric)
        return info
    '''
    def _update_file(self, global_prediction, base_predictions, meta_predictions, base_selected_idx):
        super(ArbitratedEnsembleAbstaining, self)._update_file(global_prediction, base_predictions, meta_predictions,
                                                               base_selected_idx)'''

    def get_model_name(self):
        model_name = '_'.join(['ADE', 'ABSTAIN', self.meta_error_metric.upper(), 'SIZE', str(len(self.base_models))])
        return model_name



















