from .diversity_object import *
from sklearn.feature_selection import mutual_info_regression
from diversity.information_diversity import *
from skmultiflow.utils import FastComplexBuffer, FastBuffer
import numpy as np
import pandas as pd
from itertools import combinations
from pyitlib import discrete_random_variable as drv
from scipy.stats import entropy


class SymmetricUncertainty(WindowMeasures):
	"""
	Measures symmetric uncertainty (using a sliding window) as defined in Yu - Liu paper :

	Efficient feature selection via Analysis of Relevance and Redundancy.

	We define relevance as : The symmetric uncertainty between models' predictions and the true value,
							-- high SU means high relevance. Stored as a list (size m) between models' predictions and the target
	We define redundancy as : The symmetric uncertainty between the models' predictions :
							-- high SU means high redundancy. It is stored as a matrix m x m (m : n_models)


	Input : Base models (list)
			Window size

	Methods:
		Compute_relevance_redundancy : Computes relevance and redundancy using KNN estimation of entropy
										Mutual information is measured using sklearn package : mutual_info_regression
										and entropy using the knn estimation as well
										Pros :
											Good estimation of symmetric uncertainty
										Cons :
											Not adapted for insertion, the kd-tree/ball-tree are recomputed for every instance

		Compute_relevance_redundancy_discretize : This method estimates entropy/Mutual by unspervised discretization methods
												- The values are discretized using pandas.qcut function!
												- I(X,Y) is measured as H(X) - H(X\Y)
												- Entropy is measured using scipy.stats package for discrete values

										Cons:
											This function relies on binning => Loss of information
										Pros:
											Faster than the function above

		Compute_relevance_redundancy_discretize_prime : This method is slower than the one above, it relies on pylitlib function
													"Mutual_information_normalized" that directly returns the symmetric uncertainty

	"""

	def __init__(self, model_list=None, window_size=200):
		super().__init__(model_list=model_list, window_size=window_size)
		self.true_values = FastBuffer(window_size)
		self.samples_seen = 0
		self.first_sliding = True
		self.entropy = [0.0 for _ in range(self.n_models)]

		#Measures returned
		self.redundancy = np.ones((self.n_models, self.n_models))
		self.relevance = [0 for _ in range(0, self.n_models)]

	def update_(self, predictions, true_value):
		super().update(predictions=predictions)
		self.true_values.add_element(true_value)

	def compute_redundancy(self):
		"""
		knn estimation
		:return: matrix of redundancy
		"""
		size = self.last_components_predictions.get_current_size()
		for i, j in combinations(range(self.n_models), 2):

			preds_i = [row[i] for row in self.last_components_predictions.buffer]
			preds_j = [row[j] for row in self.last_components_predictions.buffer]

			if size >= 2:
				preds_i = np.array(preds_i).reshape(-1, 1)
				mi = mutual_info_regression(X=preds_i, y=preds_j, n_neighbors= 5)
				h_i = shannon_entropy(preds_i, k=5, kth_dists=None)

				preds_j = np.array(preds_j).reshape(-1, 1)
				h_j = shannon_entropy(preds_j, k=5, kth_dists=None)

				self.redundancy[i, j] = 2 * mi[0]/(h_i + h_j)
				self.redundancy[j, i] = self.redundancy[i, j]

		redundancy_df = pd.DataFrame(self.redundancy)
		return redundancy_df

	def compute_redundancy_discretize_prime(self):
		"""
		Unsupervised dicretization
		:return:
		"""
		keys = ['preds_' + str(i) for i in range(self.n_models)]
		predictions = {key: [] for key in keys}
		for i in range(self.n_models):
			predictions['preds_'+str(i)].extend([row[i] for row in self.last_components_predictions.buffer])
		predictions_df = pd.DataFrame(predictions)
		predictions_df = predictions_df.apply(lambda s: pd.qcut(s, 10, labels=False, duplicates='drop'))

		for i, j in combinations(range(self.n_models), 2):
			preds_i = predictions_df.iloc[:,i].values.T
			preds_j = predictions_df.iloc[:, j].values.T
			mutual_information = drv.information_mutual_normalised(preds_i.tolist(), preds_j.tolist(), norm_factor='X+Y')

			self.redundancy[i, j] = mutual_information
			self.redundancy[j, i] = self.redundancy[i, j]
		return pd.DataFrame(self.redundancy)

	def compute_redundancy_discretize(self):
		"""
		Unsupervised discretization
		:return: Redundancy matrix
		"""

		for i, j in combinations(range(self.n_models), 2):
			preds_i = pd.qcut([row[i] for row in
			                   self.last_components_predictions.buffer], 10, labels=False, duplicates='drop')
			entropy_i = entropy(preds_i)
			preds_j = pd.qcut([row[j] for row in self.last_components_predictions.buffer], 10, labels=False, duplicates='drop')
			entropy_j = entropy(preds_j)
			mutual_information = entropy_i - drv.entropy_conditional(preds_i.tolist(), preds_j.tolist())
			su = mutual_information / (entropy_i + entropy_j)
			self.redundancy[i, j] = su
			self.redundancy[j, i] = self.redundancy[i, j]

		return pd.DataFrame(self.redundancy)

	#Diversity matrix is redunancy!
	def get_diversity_matrix(self, estimation_method=None):
		if estimation_method == 'knn':
			redundancy = self.compute_redundancy()
		elif estimation_method =='discretize':
			redundancy = self.compute_redundancy_discretize()
		else:
			redundancy = self.compute_redundancy_discretize_prime()
		return redundancy

	def compute_relevance_redundancy(self):
		"""
		KNN estimation
		:return: relevancy and redundancy
		"""
		keys = ['preds_'+str(i) for i in range(self.n_models)]
		predictions = {key: [] for key in keys}
		true_values = np.array(self.true_values.buffer).reshape(-1, 1)
		h_y = shannon_entropy(true_values, k=5, kth_dists=None)
		for i in range(self.n_models):
			preds_i = [row[i] for row in self.last_components_predictions.buffer]
			preds_i = np.array(preds_i).reshape(-1, 1)
			predictions['preds_'+str(i)].extend(preds_i)
			mi = mutual_info_regression(preds_i, self.true_values.buffer, n_neighbors=5)
			h_i = shannon_entropy(preds_i, k=5, kth_dists=None)
			self.entropy.append(h_i)
			SU = 2 * mi[0] / (h_i + h_y)
			self.relevance[i] = SU
			for j in range(i+1, self.n_models):
				if i < j:
					continue
				elif i == j:
					self.redundancy[i,j] = 1.0
				else:
					preds_j =np.array(predictions['preds'+str(j)]).reshape(-1, 1)
					h_j = self.entropy[j]
					self.redundancy[i, j] = mutual_info_regression(preds_i, preds_j, n_neighbors=5)/(h_i+h_j)
					self.redundancy[j, i] = self.redundancy[i, j]
		return self.relevance, pd.DataFrame(self.redundancy)

	def compute_relevance_discretize_prime(self):
		"""Unsuspervised dicretization
		:returns
		Relevancy
		"""
		true_values = pd.qcut(self.true_values.buffer, 10, labels=False, duplicates='drop')

		for i in range(self.n_models):
			pred_i = pd.qcut([row[i] for row in self.last_components_predictions.buffer], 10, labels=False, duplicates='drop')
			self.relevance[i] = drv.information_mutual_normalised(X=pred_i.tolist(), Y=true_values.tolist(), norm_factor='X+Y')
		return self.relevance

	def compute_relevance_discretize(self):
		"""
		Unsupervised dicretization : Deciles
		:return: relevancy and redundancy
		"""
		keys = ['preds_' + str(i) for i in range(self.n_models)]
		predictions = {key: [] for key in keys}
		true_values = pd.qcut(self.true_values.buffer, 10, labels=False, duplicates='drop')
		entropy_target = entropy(true_values)
		su = []
		for i in range(self.n_models):
			preds_i = pd.qcut([row[i] for row in self.last_components_predictions.buffer], 10, labels=False, duplicates='drop')
			predictions['preds_'+str(i)].extend(preds_i)
			entropy_i = entropy(preds_i)
			self.entropy.append(entropy_i)
			mutual_information = entropy_target - drv.entropy_conditional(true_values.tolist(), preds_i.tolist())
			su.append(mutual_information / (entropy_i + entropy_target))
			self.relevance[i] = su[i]

			for j in range(i+1, self.n_models):
				if i < j:
					continue
				elif i == j:
					self.redundancy[i, j] = 1.0
				else:
					mutual_info_preds = entropy_i - drv.entropy_conditional(preds_i, predictions['preds_'+str(j)])
					self.redundancy[i, j] = mutual_info_preds / (entropy_i+self.entropy[j])
					self.redundancy[j, i] = self.redundancy[i, j]
		return self.relevance, pd.DataFrame(self.redundancy)

	def get_info(self):
		return '{}:'.format(type(self).__name__) + \
		       '_ number of models :{}'.format(self.n_models) + \
		       '_measure :{}'.format('Relevancy or redundancy')

	def get_class_type(self):
		return 'window_redundancy'


