#!/usr/bin/env python

__author__ = "Dihia BOULEGANE"
__copyright__ = ""
__credits__ = ["Dihia BOULEGANE"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Dihia BOULEGANE"
__email__ = "dihia.boulegane@telecom-paristech.fr"
__status__ = "Development"

import pandas as pd


from ade.arbitrated_ensemble_abstaining import ArbitratedEnsembleAbstaining
from diversity.diversity_factory import DiversityMeasuresFactory
from skmultiflow.utils import *
from utils.functions import *
from utils.metrics import *
from ade.selection_methods import get_all_prediction_data_frame


class ArbitratedEnsembleAbstainingTradeoff(ArbitratedEnsembleAbstaining):

    #TODO: Check values of selection threshold for MMR

    def __init__(self, meta_models, base_models, meta_error_metric='MAPE', mmr_threshold=None, trade_off_lambda=0.5,
                 trade_off = 'accu_div', diversity_method='sliding_window', diversity_measure='correlation',
                 n_sliding=200, fading_factor=None, output_file=None):

        super().__init__(meta_models=meta_models, base_models=base_models, meta_error_metric=meta_error_metric,
                         output_file=output_file)

        # Threshold of the tradeoff between accuracy and diversity to be selected or no
        self.mmr_threshold = mmr_threshold
        # Trade-off parameter between accuracy and
        self.trade_off_lambda = trade_off_lambda

        # TODO: check init data
        if diversity_method == 'sliding_window':
            data = {'model_list': base_models, 'window_size': n_sliding}
        elif diversity_method == 'fading factor':
            data = {'model_list': base_models, 'alpha': fading_factor}
        elif diversity_method == 'incremental':
            data = {'model_list': base_models}
        else:
            raise NotImplementedError

        # Diversity selection parameters
        self.diversity_measure = diversity_measure

        self.n_sliding = n_sliding
        self.fading_factor = fading_factor
        div = DiversityMeasuresFactory()
        self.diversity_evaluator = div.get_diversity_evaluator(diversity_method=diversity_method, diversity_measure=diversity_measure, args_dict=data)

        if self.output_file is not None:
            super(ArbitratedEnsembleAbstainingTradeoff, self)._init_file()

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

        super(ArbitratedEnsembleAbstainingTradeoff, self).partial_fit(X, y, classes, weight)

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

            # Get Predictions of all base and meta models
            results_df, step_results = get_all_prediction_data_frame(X, self.meta_models, self.base_models)

            # Update diversity matrix with base predictions
            self.diversity_evaluator.add(step_results['all_base_predictions'])

            # Compute diversity matrix
            diversity_matrix = self.diversity_evaluator.get_diversity_matrix()

            # Rank according to predicted error ==> accuray
            results_df = results_df.sort_values(by=['meta'], ascending=True)

            # Select best predicted base model
            selected_models_df = pd.DataFrame()
            selected_models_df = pd.concat([selected_models_df, results_df.head(1)])

            # Remove selected model from results_df
            results_df = results_df.iloc[1:]

            stop = False
            while not stop:
                # Start iteration
                all_mmr_measures = []
                for idx, model in results_df.iterrows():

                    # TODO : normalize or abs correlation
                    # Compute diversity with previously selected models
                    all_correlations = [(diversity_matrix.loc[model['id']][temp_model['id']] + 1) / 2
                                        for index, temp_model in selected_models_df.iterrows()]


                    max_corr = max(all_correlations)

                    # Compute accuracy-diversity tradeoff
                    mmr = self.trade_off_lambda * (1 - model['meta']) - (1 - self.trade_off_lambda) * max_corr
                    all_mmr_measures.append((mmr, model))

                max_mmr = max(all_mmr_measures, key=lambda item: item[0])

                # TODO : add a condition on value of max mmr before adding the model
                if max_mmr[0] > self.mmr_threshold:
                    # Add model to selected committee and remove it from left models
                    selected_model = max_mmr[1].to_frame().T
                    selected_models_df = pd.concat([selected_models_df, selected_model])

                # Remove model from all results
                id_to_remove = int(max_mmr[1]['id'])

                results_df = results_df[results_df['id'] != id_to_remove]

                if results_df.empty:
                    stop = True

            # Compute final prediction based on selected models only
            if not selected_models_df.empty:
                step_results['selected_experts_idx'] = [int(i) for i in selected_models_df['id']]

                base_predictions = np.array(selected_models_df['base']).reshape(1, -1)
                weights = weight_predictions(selected_models_df['meta'])
            else:
                meta_predictions = [z.predict([X[i]])[0] for z in self.meta_models]
                base_predictions = np.array([m.predict([X[i]])[0] for m in self.base_models]).reshape(1, -1)
                weights = weight_predictions(meta_predictions)

            try:
                final_prediction = get_aggregated_result(base_predictions, weights, method='weighted_average')
                predictions.append(final_prediction)

                super(ArbitratedEnsembleAbstainingTradeoff, self)._update_outputs(global_prediction=final_prediction,
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
        return 'ArbitratedEnsembleAbstaining'

    def get_info(self):
        # TODO: et up super.info
        info = super(ArbitratedEnsembleAbstainingTradeoff, self).get_info()

        # Diversity selection parameters
        info += ' - diversity_measure: {}'.format(self.diversity_measure)
        info += ' - n_sliding: {}'.format(self.n_sliding) if self.n_sliding is not None else ''
        info += ' - fading_factor: {}'.format(self.fading_factor) if self.fading_factor is not None else ''

        # Accuracy-Diversity tradeoff parameters
        info += ' - mmr_threshold: {}'.format(self.mmr_threshold)
        info += ' - mmr_lambda: {}'.format(self.trade_off_lambda)
        return info

    def get_model_name(self):
        model_name = super(ArbitratedEnsembleAbstainingTradeoff, self).get_model_name()

        model_name = '_'.join([model_name, 'MMR', 'ACCU_DIV'])

        return model_name
