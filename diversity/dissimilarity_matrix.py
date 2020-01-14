from diversity.diversity_object import *
import pandas as pd
from itertools import combinations
from pyitlib import discrete_random_variable as drv
from scipy.stats import entropy


"""
This module implements a new measure for diversity
Dissimilarity matrix is defined as :
	D(Xi, Xj) = I(Xi;Y\Xj) + I(Xj;Y\Xi)
			  = H(Y\Xi) + H(Y\Xj) -2*H(Y\Xi,Xj)
	where :
		- I(Xi;Y\Xj) is the conditional mutual information between Xi and Xj,
		It expresses how much information the prediction Xi can predict about Y that Xj cannot

"""

#TODO : The value of dissimilarity is not restricted between 0 and 1, should be normalized


class DissimilarityMatrix(WindowMeasures):

	def __init__(self, model_list=None, window_size=200):
		super().__init__(model_list, window_size)
		self.true_values = FastBuffer(window_size)
		self.dissimilarity_matrix = np.zeros((self.n_models,self.n_models))

	def reset(self):
		super().reset()

	def update_(self, predictions, y):
		"""predictions is a list of the last predicted values from the models"""
		super().update(predictions=predictions)
		self.true_values.add_element(y)

	def get_diversity_matrix(self, estimation_method=None):
		true_values = pd.qcut(self.true_values.buffer, 10, labels=False, duplicates='drop')
		entropy_target = entropy(true_values)
		for i, j in combinations(range(self.n_models), 2):
			preds_i = [row[i] for row in self.last_components_predictions.buffer]
			preds_i_discrete = pd.qcut(preds_i, 10, labels=False, duplicates='drop')
			entropy_i_simple = entropy(preds_i_discrete)
			entropy_i = drv.entropy_conditional(true_values.tolist(), preds_i_discrete.tolist())

			preds_j = [row[j] for row in self.last_components_predictions.buffer]
			preds_j_discrete = pd.qcut(preds_j, 10, labels=False, duplicates='drop')
			entropy_j_simple = entropy(preds_j_discrete)
			entropy_j = drv.entropy_conditional(true_values.tolist(), preds_j_discrete.tolist())

			preds_ij = np.hstack((np.array(preds_i).reshape(-1,1),np.array(preds_j).reshape(-1, 1)))
			Z = np.apply_along_axis(np.array_str, axis=1, arr=preds_ij)
			Z = np.searchsorted(np.sort(Z), Z)
			entropy_ij = drv.entropy_conditional(true_values.tolist(), Z.tolist())

			self.dissimilarity_matrix[i, j] = max(0.0, (entropy_i + entropy_j - 2*entropy_ij)/(entropy_i_simple+entropy_j_simple+2*entropy_target))
			self.dissimilarity_matrix[j, i] = self.dissimilarity_matrix[i, j]
		return pd.DataFrame(self.dissimilarity_matrix)

	def get_info(self):
		return '{}:'.format(type(self).__name__) + \
		       '_ number of models :{}'.format(self.n_models) + \
		       '_measure :{}'.format('Dissimilarity')

	def get_class_type(self):
		return 'window_dissimilarity'
