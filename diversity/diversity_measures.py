from itertools import combinations
import pandas as pd

from diversity.diversity_object import *


class WindowCorrelationMeasure(WindowMeasures):
    """This class will maintain a fixed sized window of the newest
        information about base learners in the experts set, measure inside the window.
        It will measure the correlation between predictions of each model"""

    def __init__(self, model_list=None, window_size=200):
        super().__init__(model_list=model_list, window_size=window_size)

    def reset(self):
        super().reset()

    def add(self, predictions):
        """predictions is a list of the last predicted values from the models"""
        super().update(predictions=predictions)

    def get_pairwise_diversity(self, i, j):
        corr = self.get_diversity_matrix(cov=False)
        if i !=j:
            return corr.iloc[i, j]
        else:
            return 1

    def get_diversity_matrix(self, cov=False,show_plot=False, estimation_method=None):
        size = self.last_components_predictions.get_current_size()
        P = np.zeros((size, self.n_models))
        P = pd.DataFrame(P)
        corr = np.zeros((self.n_models, self.n_models))
        corr = pd.DataFrame(corr)
        for i in range(self.n_models):
            P.iloc[:, i] = [row[i] for row in self.last_components_predictions.buffer]
        if size > 1:
            if cov:
                corr = P.cov()
            else:
                corr = P.corr()
        return corr

    def get_avg_corr(self, ensemble_size):
        sum_ens = 0
        count = 0
        for i in range(0, ensemble_size):
            for j in range(i+1, ensemble_size):
                sum_ens += self.get_pairwise_diversity(i, j)
                count += 1
        return sum_ens/count

    def get_info(self):
        return '{}:'.format(type(self).__name__) + \
               '_ number of models :{}'.format(self.n_models) + \
               '_measure :{}'.format('Correlation')

    def get_class_type(self):
        return 'window_correlation'


class WindowEntropyMeasure(WindowCorrelationMeasure):
    """Assume that distribution is gaussian :
      I(X,Y) = -.5*log(1-correlation(X,Y)^2)"""

    def __init__(self, model_list=None, window_size=200):
        super().__init__(model_list=model_list, window_size=window_size)

    def reset(self):
        super().reset()

    def add(self, predictions=None):
        """predictions is a list of the last predicted values from the models
            """
        super().update(predictions=predictions)

    def get_pairwise_diversity(self, i, j):
        corr_ij = super().get_pairwise_diversity(i, j)
        rho_ij = -.5* np.log(1 - 0.99**2)
        if corr_ij != 1:
            rho_ij = -.5 * np.log(1 - corr_ij ** 2)
        return rho_ij

    def get_diversity_matrix(self, cov=False, show_plot=False):
        corr = super().get_diversity_matrix()
        rho = np.where(np.array(corr.values) == 1, 0.9, np.array(corr.values))
        entropy = -.5 * np.log(1 - rho ** 2)
        entropy = pd.DataFrame(entropy)
        return entropy

    def get_info(self):
        return '{}:'.format(type(self).__name__) + \
               '_ number of models :{}'.format(self.n_models) + \
               '_measure :{}'.format('Entropy')

    def get_class_type(self):
        return 'window_entropy'


class DisagreementEvaluator(FadingFactorMeasures):
    """Incremental -with fading- measure of pairwise double fault, and disagreement """

    def __init__(self, model_list, alpha):
        super().__init__(model_list=model_list, alpha=alpha)
        self.last_ensemble_prediction = FastBuffer(1)
        self.binary_values_ens = [0 for i in range(self.n_models)]
        self.estimator_ens = np.zeros((self.n_models, self.n_models))

    def add(self, predictions, model_list=None, y_true=None):
        old_predictions = self.last_predicted_values.add_element(predictions)
        old_true_value = self.last_true_value.add_element(y_true)
        if (old_predictions is not None) and (old_true_value is not None):
            self.binarize(y_true[0], predictions)
            for i, j in combinations(model_list, 2):
                self.estimator[i, j] = self.alpha * self.estimator[i, j] + self.disagreement(self.binary_values[i],
                                                                                             self.binary_values[j])
        self.increment = self.alpha * self.increment + 1.0

    def add_ens(self, predictions, model_list=None, y_ens=None):
        # Disagreement with the ensemble
        self.last_predicted_values.add_element(predictions)
        self.last_ensemble_prediction.add_element(y_ens)
        self.binarize_ens(y_ens[0], predictions)

        for i, j in combinations(model_list, 2):
            self.estimator_ens[i, j] = self.alpha * self.estimator_ens[i, j] + self.disagreement(self.binary_values_ens[i],
                                                                                                 self.binary_values_ens[j])
        self.increment = self.alpha * self.increment + 1.0

    # Disagreement between models
    def binarize(self, y_true, predictions):
        super().binarize(y_true=y_true, predictions=predictions)

    # Disagreement between the ensemble:
    def binarize_ens(self, y_ens, predictions):
        epsilon = 1
        for i in range(self.n_models):
            if np.abs(predictions[i]-y_ens) <= epsilon:
                self.binary_values_ens[i] = 1
            else:
                self.binary_values_ens[i] = 0

    def get_disagreement(self, ensemble_size):
        sum_ens = 0
        count = 0
        for i in range(0, ensemble_size):
            for j in range(i+1, ensemble_size):
                #Disagreement between models
                sum_ens += self.get_pairwise_diversity(i, j)

                count += 1
        return sum_ens/count

    def get_disagreement_ens(self, ensemble_size):
        sum_ens = 0
        count = 0
        for i in range(0, ensemble_size):
            for j in range(i+1, ensemble_size):
                #Models vs predictions
                sum_ens += self.get_pairwise_diversity_ens(i, j)
                count += 1
        return sum_ens/count

    def get_pairwise_diversity(self, i, j):
        if self.increment > 0.0:
            return self.estimator[i, j] / self.increment
        else:
            return 0.0

    def get_pairwise_diversity_ens(self, i, j):
        if self.increment > 0.0:
            return self.estimator_ens[i, j] / self.increment
        else:
            return 0.0

    def disagreement(self, i, j):
        return self.diversity_contingency(i, j, 0, 1) + self.diversity_contingency(i, j, 1, 0)

    # Return a diversity matrix
    def get_diversity_matrix(self):
        K = self.n_models
        div = np.empty((K, K), dtype=float)
        div = pd.DataFrame(div)
        for i in range(K):
            for j in range(K):
                if i > j:
                    continue
                if i == j:
                    value = 0.0
                else:
                    value = self.get_pairwise_diversity(i, j)
                div.iloc[i, j] = value
                div.iloc[j, i] = value
        div.columns = range(K)
        div.index = range(K)
        return div

    def get_info(self):
        return '{}:'.format(type(self).__name__) + \
               '_ number of models :{}'.format(self.n_models) + \
               '_measure :{}'.format('disagreement')

    def get_class_type(self):
        return 'ff_disagreement'


class DoubleFaultEvaluator(FadingFactorMeasures):
    """Incremental -with fading- measure of pairwise double fault """

    def __init__(self, model_list, alpha):
        super().__init__(model_list=model_list, alpha=alpha)

    def add(self, predictions, model_list=None, y_true=None):
        old_predictions = self.last_predicted_values.add_element(predictions)
        old_true_value = self.last_true_value.add_element(y_true)
        if (old_predictions is not None) and (old_true_value is not None):
            self.binarize(old_true_value[0], old_predictions)
            for i, j in combinations(model_list, 2):
                self.estimator[i, j] = self.alpha * self.estimator[i, j] + self.double_fault(self.binary_values[i], self.binary_values[j])
            self.increment = self.alpha * self.increment + 1.0

    def binarize(self, y_true, predictions):
        super().binarize(y_true=y_true, predictions=predictions)

    def get_pairwise_diversity(self, i, j):
        if self.increment > 0.0:
            return self.estimator[i, j] / self.increment
        else:
            return 0.0

    def double_fault(self, i, j):
        return self.diversity_contingency(i, j, 0, 0)

    # Return a diversity matrix
    def get_diversity_matrix(self):
        K = self.n_models
        div = np.empty((K, K), dtype=float)
        div = pd.DataFrame(div)
        for i in range(K):
            for j in range(K):
                if i > j:
                    continue
                if i == j:
                    value = 0.0
                else:
                    value = self.get_pairwise_diversity(i, j)
                div.iloc[i, j] = value
                div.iloc[j, i] = value
        div.columns = range(K)
        div.index = range(K)
        return div

    def get_info(self):
        return '{}:'.format(type(self).__name__) + \
               '_ number of models :{}'.format(self.n_models) + \
               '_measure :{}'.format('double_fault')

    def get_class_type(self):
        return 'ff_double_fault'


class WindowDisagreementEvaluator(WindowClassificationMeasures):
    """Pairwise diversity-double fault on a sliding window """

    def __init__(self, model_list, window_size=200):
        super().__init__(model_list=model_list, window_size=window_size)
        self.index = 0
        self.samples_seen = 0
        self.first_sliding = True

    def add(self, predictions, model_list=None, y_true=None):
        self.last_components_predictions.add_element(predictions)
        size = self.last_components_predictions.get_current_size()
        self.last_true_value.add_element(y_true)
        if self.first_sliding:
            self.first_sliding = False
            self.samples_seen = size
            for pos in range(size):
                self.binarize(self.last_true_value.buffer[pos], self.last_components_predictions.buffer[pos])
                for i, j in combinations(model_list, 2):
                    self.estimator[i, j] += self.disagreement(self.binary_values[i][pos], self.binary_values[j][pos])
        else:
            self.samples_seen += 1
            if 0 < size <= self.window_size:
                self.index = size
            else:
                self.index = (self.index % 200) + 1
            pos = self.index - 1
            self.binarize(self.last_true_value.buffer[pos], self.last_components_predictions.buffer[pos])
            for i, j in combinations(model_list, 2):
                self.estimator[i, j] += self.disagreement(self.binary_values[i][pos], self.binary_values[j][pos])

    def get_disagreement(self, ensemble_size):
        sum_ens = 0
        count = 0
        for i in range(0, ensemble_size):
            for j in range(i+1, ensemble_size):
                sum_ens += self.get_pairwise_diversity(i, j)
                count += 1
        return sum_ens/count

    def binarize(self, y_true, predictions):
        return super().binarize(y_true=y_true, predictions=predictions)

    def get_pairwise_diversity(self, i, j):
        if self.samples_seen > 0:
            return self.estimator[i, j] / self.samples_seen
        else:
            return 0.0

    def disagreement(self, i, j):
        return self.diversity_contingency(i, j, 0, 1) + self.diversity_contingency(i, j, 1, 0)

    # Return a diversity matrix
    def get_diversity_matrix(self, model_list=None, show_plot=False):
        K = self.n_models
        div = np.empty((K, K), dtype=float)
        div = pd.DataFrame(div)
        for i in range(K):
            for j in range(K):
                if i > j:
                    continue
                if i == j:
                    value = 0.0
                else:
                    value = self.get_pairwise_diversity(i, j)
                div.iloc[i, j] = value
                div.iloc[j, i] = value
        div.columns = range(K)
        div.index = range(K)
        return div


    def get_info(self):
        return '{}:'.format(type(self).__name__) + \
               '_ number of models :{}'.format(self.n_models) + \
               '_measure :{}'.format('double fault and disagreement')

    def get_class_type(self):
        return 'window_disagree'


class WindowDoubleFaultEvaluator(WindowClassificationMeasures):
    """Pairwise diversity-double fault on a sliding window """

    def __init__(self, model_list, window_size=200):
        super().__init__(model_list=model_list, window_size=window_size)
        self.index = 0
        self.samples_seen = 0
        self.first_sliding = True

    def add(self, predictions, model_list=None, y_true=None):
        self.last_components_predictions.add_element(predictions)
        size = self.last_components_predictions.get_current_size()
        self.samples_seen = size
        self.last_true_value.add_element(y_true)
        if self.first_sliding:
            self.first_sliding = False
            self.samples_seen = size
            for pos in range(size):
                self.binarize(self.last_true_value.buffer[pos], self.last_components_predictions.buffer[pos])
                for i, j in combinations(model_list, 2):
                    self.estimator[i, j] += self.double_fault(self.binary_values[i][pos], self.binary_values[j][pos])
        else:
            self.samples_seen += 1
            if 0 < size <= self.window_size:
                self.index = size
            else:
                self.index = (self.index % 200) + 1
            pos = self.index - 1
            self.binarize(self.last_true_value.buffer[pos], self.last_components_predictions.buffer[pos])
            for i, j in combinations(model_list, 2):
                self.estimator[i, j] += self.double_fault(self.binary_values[i][pos], self.binary_values[j][pos])

    def get_doublefault(self, ensemble_size):
        sum_ens = 0
        count = 0
        for i in range(0, ensemble_size):
            for j in range(i+1, ensemble_size):
                sum_ens += self.get_pairwise_diversity(i, j)
                count += 1
        return sum_ens/count

    def binarize(self, y_true, predictions):
        return super().binarize(y_true=y_true, predictions=predictions)

    def get_pairwise_diversity(self, i, j):
        if self.samples_seen > 0:
            return self.estimator[i, j] / self.samples_seen
        else:
            return 0.0

    def double_fault(self, i, j):
        return self.diversity_contingency(i, j, 0, 0)

    # Return a diversity matrix
    def get_diversity_matrix(self, model_list=None, show_plot=None):
        K = self.n_models
        div = np.empty((K, K), dtype=float)
        div = pd.DataFrame(div)
        for i in range(K):
            for j in range(K):
                if i > j:
                    continue
                if i == j:
                    value = 0.0
                else:
                    value = self.get_pairwise_diversity(i, j)
                div.iloc[i, j] = value
                div.iloc[j, i] = value
        div.columns = range(K)
        div.index = range(K)
        return div

    def get_info(self):
        return '{}:'.format(type(self).__name__) + \
               '_ number of models :{}'.format(self.n_models) + \
               '_measure :{}'.format('double fault and disagreement')

    def get_class_type(self):
        return 'window_double_fault'


class IncrementalDisagree(DisagreementEvaluator):
    def __init__(self, model_list):
        super().__init__(model_list=model_list, alpha=1.0)

    def add(self, predictions, model_list=None, y_true=None):
        super().add(predictions=predictions, model_list=model_list, y_true=y_true)

    def binarize(self, y_true, predictions):
        return super().binarize(y_true=y_true, predictions=predictions)

    def get_pairwise_diversity(self, i, j):
        return super().get_pairwise_diversity(i=i, j=j)

    def disagreement(self, i, j):
        return super().disagreement(i=i, j=j)

    def get_diversity_matrix(self):
        return super().get_diversity_matrix()

    def get_disagreement(self, ensemble_size):
        sum_ens = 0
        count = 0
        for i in range(0, ensemble_size):
            for j in range(i+1, ensemble_size):
                sum_ens += self.get_pairwise_diversity(i, j)
                count += 1
        return sum_ens/count

    def get_info(self):
        return '{}:'.format(type(self).__name__) + \
               '_ number of models :{}'.format(self.n_models) + \
               '_measure :{}'.format('disagreement')

    def get_class_type(self):
        return 'inc_disagreement'


class IncrementalDoubleFault(DoubleFaultEvaluator):
    def __init__(self, model_list):
        super().__init__(model_list=model_list, alpha=1.0)

    def add(self, predictions, model_list=None, y_true=None):
        super().add(predictions=predictions, model_list=model_list, y_true=y_true)

    def binarize(self, y_true, predictions):
        return super().binarize(y_true=y_true, predictions=predictions)

    def get_pairwise_diversity(self, i, j):
        return super().get_pairwise_diversity(i=i, j=j)

    def disagreement(self, i, j):
        return super().double_fault(i=i, j=j)

    def get_diversity_matrix(self):
        return super().get_diversity_matrix()

    def get_doublefault(self, ensemble_size):
        sum_ens = 0
        count = 0
        for i in range(0, ensemble_size):
            for j in range(i+1, ensemble_size):
                sum_ens += self.get_pairwise_diversity(i, j)
                count += 1
        return sum_ens/count

    def get_info(self):
        return '{}:'.format(type(self).__name__) + \
                '_ number of models :{}'.format(self.n_models) + \
                '_measure :{}'.format('disagreement')

    def get_class_type(self):
        return 'inc_double_fault'
