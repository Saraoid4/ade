#!/usr/bin/env python
__author__ = "Dihia BOULEGANE"
__copyright__ = ""
__credits__ = ["Dihia BOULEGANE"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Dihia BOULEGANE"
__email__ = "dihia.boulegane@telecom-paristech.fr"
__status__ = "Development"

import heapq
import numpy as np
import pandas as pd 
from sklearn.preprocessing import normalize
from itertools import product


from utils.metrics import compute_mape

# TODO: Review foftmax function
def softmax(values):
    """ softmax

    Computes the softmax of x based on the values

    Parameters
    ----------
    x: Array-like
    values : ndarray of float
        Array containing the predicted MSE for each base model that is part of the expert subset

    Returns
    -------
    The softmax of x computed based on all items of values

    """
    try:
        normalized_predictions = normalize(np.array(values).reshape(1, -1))

        inverse_normalized_predictions = [-1 * v for v in normalized_predictions]

        sft_max_normalized = np.exp(inverse_normalized_predictions) / np.sum(np.exp(inverse_normalized_predictions))

        return sft_max_normalized.reshape(1, -1)
    except Exception as exc:
        print(exc)


def weight_predictions(all_predicted_errors, method='softmax'):
    """ Assigns a weight to  a base model according to its predicted error according to predicted errors of
        all base models. The higher the error, the lower the weight

    Parameters
    ----------

    all_predicted_errors : array of float
        Array containing the predicted MSE for each base model that is part of the expert subset
    method : string
        Name of the method that will be used to compute the weight
    Returns
    -------
    A weight to control the contribution of the base model in the final aggregated prediction

    """

    if method == 'softmax':
        w = softmax(all_predicted_errors)
        return w
    else:
        raise NotImplementedError


def get_aggregated_result(predictions, weights, method='weighted_average'):

    if method == 'weighted_average':
        return weighted_average(predictions, weights)
    elif method == 'simple_average':
        return np.average(predictions)
    else:
        raise NotImplementedError


def weighted_average(values, weights):
    """ weighted_average

    Parameters
    ----------
    values: ndarray
    weights: ndarray

    Returns
    -------
    The weighted average
    """
    if np.sum(weights) == 0:
        return np.average(values)
    return np.average(values, weights=weights)


def simple_average(predictions):
    return np.average(predictions)

# Same functions as above except they handle dictionaries to keep a trace of the models

def softmax_2(dicto):
    """ softmax

    Computes the softmax of x based on the values

    Parameters
    ----------
    x: Array-like
    values : Dictionnary

    Returns
    -------
    The softmax of x computed based on all items of values

    """
    sft_max_normalized = {}
    try:
        values = list(dicto.values())

        normalized_predictions = normalize(np.array(values).reshape(1, -1))

        inverse_normalized_predictions = [-1 * v for v in normalized_predictions]

        for i, (key, value) in enumerate(dicto.items()):

            sft_max_normalized[key] = np.exp(inverse_normalized_predictions[0][i])/np.sum(np.exp(inverse_normalized_predictions))
        return sft_max_normalized
    except Exception as exc:
        print(exc)


def weight_predictions_2(dicto, method='softmax'):
    """ Assigns a weight to  a base model according to its predicted error according to predicted errors of
        all base models. The higher the error, the lower the weight

    Parameters
    ----------

    dicto : dictionary
    method : string
        Name of the method that will be used to compute the weight
    Returns
    -------
    A weight to control the contribution of the base model in the final aggregated prediction

    """

    if method == 'softmax':
        w = softmax_2(dicto)
        return w
    else:
        raise NotImplementedError

#Algorithm in Arbitrage forecasting experts


def re_estimate_weights(meta_predictions, base_models, diversity_evaluator):

    final_weights = {}

    weights_dict = weight_predictions_2(meta_predictions)
    key_b, value_b = heapq.nlargest(1, weights_dict.items())[0]

    # Initialize the list with final weights
    final_weights[key_b] = value_b
    del base_models[key_b]

    for i, m in base_models.items():
        final_weights[i] = weights_dict[i]
        for j in final_weights.keys():
            corr = diversity_evaluator.get_pairwise_diversity(i, j)


            penalty = corr * final_weights[j] * final_weights[i]

            final_weights[j] += penalty
            final_weights[i] -= penalty

    weights = np.array(list(final_weights.values())).reshape(1, -1)
    return weights

def compute_number_of_values_per_unit(relaibility_matrix):
    """ computes the number of non missing values per unit in the reliability matrix
    
    Parameters
    ----------
    relaibility_matrix : 2-D array
        each column stands for a unit to be coded and each row for a rater (coder, voter ...)
    
    Returns
    -------
    list of int
        number of non missing values per unit
    """

    values_per_unit = list(relaibility_matrix.count())

    return values_per_unit

def compute_algebraic_distance(y_1, y_2):
    # return ((y_1 - y_2) / (y_1 + y_2))**2
    return (y_1 - y_2)**2

def compute_distance_matrix(relaibility_matrix):
    # All values will contain all unique values of outputs and stands for units in krippendorff computation 
    all_values = set()
    [all_values.update(relaibility_matrix[i].values) for i in relaibility_matrix.columns]
    all_values = sorted(all_values)

    # Init distance matrix to zeros 2D array including all possible pairs
    distance_df = pd.DataFrame({i:[0.0]* len(all_values) for i in all_values}, index = all_values)

    # Update distances 
    for c in distance_df.columns:
        for i in list(distance_df.index):
            # TODO: make it simple without calling a function for time efficiency
            distance_df.at[i, c] = compute_algebraic_distance(i, c)
    return distance_df

def compute_disagreement(relaibility_matrix):
    # Delete columns where only one selected
    for c in relaibility_matrix.columns:
        if relaibility_matrix[c].count() == 1:
            del relaibility_matrix[c]

    # Compute distance matrix between predictions
    distance_matrix = compute_distance_matrix(relaibility_matrix.dropna())
    
    # distance_matrix = distance_matrix.dropna()
    nb_entry_per_entity = relaibility_matrix.count()

    weighted_vote_per_distance = []
    marginal_weighted_vote = []
    for o_1, o_2 in product(distance_matrix.columns, distance_matrix.columns):
        delta = distance_matrix.at[o_1, o_2]
        # iterate over columns
        values_per_entity = []

        total_count_o_1 = 0
        total_count_o_2 = 0

        for e in relaibility_matrix.columns:
            nb_o_1_per_e = relaibility_matrix[e].loc[relaibility_matrix[e] == o_1].count()
            total_count_o_1 += nb_o_1_per_e

            nb_o_2_per_e = relaibility_matrix[e].loc[relaibility_matrix[e] == o_2].count()
            values_per_entity.append((nb_o_1_per_e*nb_o_2_per_e)/ (nb_entry_per_entity[e] - 1))
            total_count_o_2 += nb_o_2_per_e


        weighted_vote_per_distance.append(delta * np.sum(values_per_entity))
        # disagreement by chance
        marginal_weighted_vote.append(delta * total_count_o_1 * total_count_o_2)    
    dis_exp = 1 / np.sum(nb_entry_per_entity) *  np.sum(marginal_weighted_vote)

     
    dis_obs = 1/ np.sum(nb_entry_per_entity) * np.sum(weighted_vote_per_distance)

    return dis_obs, dis_exp 

def compute_agreement(reliability_matrix):

    dis_obs, dis_exp = compute_disagreement(reliability_matrix)

    if not dis_exp:
        # Zero chance of disagrement
        return 1
    else:
        # print()
        return 1 - (dis_obs / dis_exp)
