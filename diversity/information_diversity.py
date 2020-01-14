#source : pyBILT

"""Functions to evaluate information theoretic measures using knn approaches.

This module defines a set of functions to compute information theoretic
measures (i.e. Shannon Entropy, Mutual Information, etc.) using the
k-nearest neighbors (knn) approach.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.special import gamma, psi
from scipy.spatial.distance import cdist
from six.moves import range
import numpy as np

def k_nearest_neighbors(X, k=1):
	"""Get the k-nearest neighbors between points in a random variable/vector.

	Determines the k nearest neighbors for each point in teh random
	variable/vector using Euclidean style distances.

	Args:
		X (np.array): A random variable of form X = {x_1, x_2, x_3,...,x_N} or
			a random vector of form X = {(x_1, y_1), (x_2, y_2),...,(x_N,
			y_n)}.
		k (Optional[int]): The number of nearest neighbors to store for each
			point. Defaults to 1.

	Returns:
		dict: A dictionary keyed by the indices of X and containing a list
			of the k nearest neighbor for each point along with the distance
			value between the point and the nearest neighbor.
	"""
	nX = len(X)
	# initialize knn dict
	knn = {key: [] for key in range(nX)}
	# make sure X has the right shape for the cdist function
	X = np.reshape(X, (nX,-1))
	dists_arr = cdist(X, X)
	distances = [[i,j,dists_arr[i,j]] for i in range(nX-1) for j in range(i+1,nX)]
	# sort distances
	distances.sort(key=lambda x: x[2])
	# pick up the k nearest
	for d in distances:
		i = d[0]
		j = d[1]
		dist = d[2]
		if len(knn[i]) < k:
			knn[i].append([j, dist])
		if len(knn[j]) < k:
			knn[j].append([i, dist])
	return knn



# knn kth neighbor distances for entropy calcs.

def kth_nearest_neighbor_distances(X, k=1):
	"""Returns the distance for the kth nearest neighbor of each point.

   Args:
		X (np.array): A random variable of form X = {x_1, x_2, x_3,...,x_N} or
			a random vector of form X = {(x_1, y_1), (x_2, y_2),...,(x_N,
			y_n)}.
		k (Optional[int]): The number of nearest neighbors to check for each
			point. Defaults to 1.
	Returns:
		list: A list in same order as X with the distance value to the kth
		nearest neighbor of each point in X.

	"""
	nX = len(X)
	# make sure X has the right shape for the cdist function
	X = np.reshape(X, (nX,-1))
	dists_arr = cdist(X, X)
	# sorts each row
	dists_arr.sort()
	return [dists_arr[i][k] for i in range(nX)]



def shannon_entropy(X, k=1, kth_dists=None):
	"""Return the Shannon Entropy of the random variable/vector.

	This function computes the Shannon information entropy of the
	random variable/vector as estimated using the Kozachenko-Leonenko (KL)
	knn estimator.

	Args:
		X (np.array): A random variable of form X = {x_1, x_2, x_3,...,x_N} or
			a random vector of form X = {(x_1, y_1), (x_2, y_2),...,(x_N,
			y_n)}.
		k (Optional[int]): The number of nearest neighbors to store for each
			point. Defaults to 1.
		kth_dists (Optional[list]): A list in the same order as points in X
			that has the pre-computed distances between the points in X and
			their kth nearest neighbors at. Defaults to None.

	References:
		1. Damiano Lombardi and Sanjay Pant, A non-parametric k-nearest
			neighbour entropy estimator, arXiv preprint,
				[cs.IT] 2015, arXiv:1506.06501v1.
				https://arxiv.org/pdf/1506.06501v1.pdf

		2. https://www.cs.tut.fi/~timhome/tim/tim/core/differential_entropy_kl_details.htm

		3. Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
			of a random vector. Probl. Inf. Transm. 23, 95-101.

	Returns:
		float: The estimate of the Shannon Information entropy of X.

	"""
	# the kth nearest neighbor distances
	r_k = kth_dists
	if kth_dists is None:
		r_k = np.array(kth_nearest_neighbor_distances(X, k=k))
	r_k = np.where(r_k==0, 1e-06, r_k)
	# length
	n = len(X)
	# dimension
	d = 1
	if len(X.shape) == 2:
		d = X.shape[1]
	# volume of unit ball in d^n
	v_unit_ball = np.pi**(0.5*d)/gamma(0.5*d + 1.0)
	# log distances
	lr_k = np.log(r_k)
	# Shannon entropy estimate
	H = psi(n) - psi(k) + np.log(v_unit_ball) + (np.float(d)/np.float(n))*(lr_k.sum())
	return H



