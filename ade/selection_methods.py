import sys
import math
import pandas as pd
from sklearn import preprocessing
from utils.functions import *



"""
def select_n_best(meta_models, base_models, X, n_best):
    # TODO: Maybe change return policy
    sub_meta_models = []
    sub_base_models = []
    sub_meta_predictions = []
    sub_base_predictions = []

    step_results = {'all_meta_predictions': [], 'all_base_predictions': [], 'selected_experts_idx': []}

    # Store all predictions
    for i in range(len(meta_models)):
        step_results['all_base_predictions'].append(meta_models[i].predict(X)[0])
        step_results['all_base_predictions'].append(base_models[i].predict(X)[0])

    # Select n_best base models only
    meta_series = pd.Series(step_results['all_meta_predictions'])
    # Rank by decreasing order
    meta_series = meta_series.sort_values(ascending=True)
    sub_experts = meta_series.head(n_best)
    step_results['selected_experts_idx'] = list(sub_experts.index)

    for i in step_results['selected_experts_idx']:
        sub_meta_models.append(meta_models[i])
        sub_base_models.append(base_models[i])

        sub_meta_predictions.append(meta_models[i].predict(X)[0])
        sub_base_predictions.append(base_models[i].predict(X)[0])

    return sub_meta_models, sub_base_models, sub_meta_predictions, sub_base_predictions, step_results"""


def select_experts_threshold(meta_models, base_models, X, threshold_accu):
    # TODO: Maybe change return policy
    sub_meta_models = []
    sub_base_models = []
    sub_meta_predictions = []
    sub_base_predictions = []

    step_results = {'all_meta_predictions': [], 'all_base_predictions': [], 'selected_experts_idx': []}

    for i in range(len(meta_models)):
        z = meta_models[i]
        # Getting and storing meta prediction
        predicted_error = z.predict(X)[0]
        step_results['all_meta_predictions'].append(predicted_error)

        # Getting and storing base prediction
        step_results['all_base_predictions'].append(base_models[i].predict(X)[0])

        if predicted_error <= threshold_accu:
            sub_meta_models.append(z)
            sub_base_models.append(base_models[i])
            sub_meta_predictions.append(predicted_error)
            sub_base_predictions.append(base_models[i].predict(X)[0])
            # Storing expert index
            step_results['selected_experts_idx'].append(i)

    return sub_meta_models, sub_base_models, sub_meta_predictions, sub_base_predictions, step_results


def select_relevant_models(base_models, X, relevance_threshold, symmetric_uncertainty):

    # TODO: Maybe change return policy
    sub_base_models = []
    sub_base_predictions = []
    step_results = {'all_relevance': [], 'all_base_predictions': [], 'selected_experts_idx': []}

    step_results['all_base_predictions'].extend([m.predict(X)[0] for m in base_models])

    #Compute relevance of the models
    relevance = symmetric_uncertainty.compute_relevance()
    step_results['all_relevance'].extend(relevance)
    relevance_df = pd.DataFrame(relevance, columns=['relevance'])

    #Select most relevant models
    relevant_models = relevance_df[relevance_df['relevance']>=relevance_threshold]

    if not relevant_models.empty:
        #Sorted list in descending order by relevance:
        relevant_models.sort_values(by=['relevance'], ascending=False)
        relevant_models_idx = relevant_models.index.tolist()

        #Most relevant model index:
        best_idx = relevant_models_idx[0]
        sub_base_models.append(base_models[best_idx])
        sub_base_predictions.append(base_models[best_idx].predict(X)[0])
        step_results['selected_experts_idx'].append(best_idx)

        relevant_models_idx = relevant_models_idx[1:]
        if relevant_models_idx:
            redundancy = symmetric_uncertainty.compute_redundancy()
            for j in relevant_models_idx:
                if redundancy.loc[best_idx][j] <= relevant_models.loc[j]['relevance']:
                    sub_base_models.append(base_models[j])
                    sub_base_predictions.append(base_models[j].predict(X)[0])
                    step_results['selected_experts_idx'].append(j)

    return sub_base_models, sub_base_predictions, step_results


def select_experts_prob(meta_models, base_models, X, meta_confidence_level):
    sub_meta_models = []
    sub_base_models = []
    sub_meta_predictions = []
    sub_base_predictions = []

    step_results = {'all_meta_predictions': [], 'all_base_predictions': [], 'selected_experts_idx': []}

    for i in range(len(meta_models)):
        z = meta_models[i]
        # TODO: handle negative errors predicted
        predicted_error = z.predict(X)[0]
        step_results['all_meta_predictions'].append(predicted_error)

        # Getting and storing base prediction
        step_results['all_base_predictions'].append(base_models[i].predict(X)[0])

        # random bernoulli on confidence
        if meta_confidence_level:
            random_conf = np.random.binomial(1, z.get_confidence_score(), 1)
        else:
            random_conf = 1

        # random bernoulli inverse predicted error
        p = 1 - np.abs(predicted_error) / 100
        if p > 1 or p < 0:
            random_error = np.random.binomial(1, sys.float_info.epsilon, 1)
        else:
            random_error = np.random.binomial(1, p, 1)[0]
        if random_conf * random_error:
            # select base model
            sub_meta_models.append(z)
            sub_base_models.append(base_models[i])
            sub_meta_predictions.append(predicted_error)
            sub_base_predictions.append(base_models[i].predict(X)[0])
            step_results['selected_experts_idx'].append(i)

    return sub_meta_models, sub_base_models, sub_meta_predictions, sub_base_predictions, step_results


def select_experts_diversity(meta_models, base_models, X, diversity_evaluator, diversity_measure, meta_confidence_level, threshold_accu, threshold_div, n_best):

    sub_meta_models = {}
    sub_base_models = {}
    sub_meta_predictions = {}
    sub_base_predictions = {}

    sub_base_diverse_predictions = {}
    sub_diverse_models = {}

    step_results = {'all_meta_predictions': [], 'all_base_predictions': [], 'selected_experts_idx': []}

    # dictionary of meta-base models
    meta_models_d = {i: meta_models[i] for i in range(0, len(meta_models))}

    if diversity_evaluator is not None and (diversity_measure == 'correlation' or diversity_measure == 'entropy'):
        diversity_evaluator.add(get_all_predictions(base_models, X))

    for i, (name, z) in enumerate(meta_models_d.items()):
        predicted_error = z.predict(X)[0]

        # storing all meta_predictions
        step_results['all_meta_predictions'].append(predicted_error)
        # Storing all base_predictions
        step_results['all_base_predictions'].append(base_models[i].predict(X)[0])
        if meta_confidence_level:
            conf = z.get_confidence_score()
        else:
            conf = 1
        predicted_error = predicted_error * conf

        if np.abs(predicted_error) <= threshold_accu:
            sub_meta_models[name] = z
            sub_base_models[name] = base_models[i]
            sub_meta_predictions[name] = predicted_error
            sub_base_predictions[name] = base_models[name].predict(X)[0]

    """Step of selection by diversity"""
    # Check if sub_base_models is empty
    if sub_base_models:
        # Seek n_best models
        heap = [(abs(value), key) for key, value in sub_meta_predictions.items()]
        smallest = heapq.nsmallest(n_best, heap)
        best_models_indexes = [key for value, key in smallest]

        # store idx of seed sub set of experts
        for idx in best_models_indexes:
            step_results['selected_experts_idx'].append(idx)

        # Initialize diversity dict
        sub_diverse_models = {key: sub_base_models[key] for key in best_models_indexes}
        sub_base_diverse_predictions = {key: sub_base_predictions[key] for key in best_models_indexes}
        n_best_models = list(sub_diverse_models.keys())

        #ranking approach
        if diversity_evaluator is not None and (diversity_measure == 'correlation' or diversity_measure == 'entropy'):
            potential_models = list(set(sub_base_models.keys()) - set(n_best_models))
            if len(potential_models) > 0:
                diversity_matrix = get_experts_diversity(diversity_evaluator, sub_base_models)
                index_to_add, index_to_remove = get_index(diversity_matrix, potential_models, n_best_models, threshold_div)
                for j in index_to_add:
                    sub_diverse_models[j] = sub_base_models[j]
                    sub_base_diverse_predictions[j] = sub_base_models[j].predict(X)[0]
                    step_results['selected_experts_idx'].append(j)
                for k in index_to_remove:
                    del sub_meta_models[k]
                    del sub_meta_predictions[k]
        #Naive approach
        if diversity_measure == 'double_fault' or diversity_measure == 'disagree' or diversity_measure == 'dissimilarity':
            diversity_matrix = get_experts_diversity(diversity_evaluator, sub_base_models)
            for i in sorted(sub_diverse_models.keys()):
                index_to_add = get_div_index(diversity_matrix, diversity_measure, i, threshold_div)
                for j in index_to_add:
                    sub_diverse_models[j] = sub_base_models[j]
                    sub_base_diverse_predictions[j] = sub_base_models[j].predict(X)[0]
                    step_results['selected_experts_idx'].append(j)

    # convert dictionaries to lists
    final_sub_diverse_models = list(sub_diverse_models.values())
    final_sub_base_diverse_predictions = list(sub_base_diverse_predictions.values())

    final_sub_meta_models = [sub_meta_models[key] for key in sub_diverse_models.keys()]
    final_sub_meta_predictions = [sub_meta_predictions[key] for key in sub_diverse_models.keys()]
    return final_sub_meta_models, final_sub_diverse_models, final_sub_meta_predictions, final_sub_base_diverse_predictions, step_results


def select_experts_prob_diversity(meta_models, base_models, X, diversity_evaluator, diversity_measure, meta_confidence_level, threshold_div, n_best):
    """
    Selects subset of base models as experts based on the predictions of the meta models
       Parameters
       ----------
       X: Array-like
           Test instance on which will be based the selection of experts

       Returns
       -------
       # TODO: Document return values
       Tuple :
    """
    sub_meta_models = {}
    sub_base_models = {}
    sub_meta_predictions = {}
    sub_base_predictions = {}

    sub_base_diverse_predictions = {}
    sub_diverse_models = {}

    step_results = {'all_meta_predictions': [], 'all_base_predictions': [], 'selected_experts_idx': []}

    # dictionary of meta models
    meta_models_d = {i: meta_models[i] for i in range(0, len(meta_models))}

    # adds a list of the last predicted values to the buffer
    if diversity_measure is not None and (diversity_measure == 'correlation' or diversity_measure == 'entropy'):
        diversity_evaluator.add(get_all_predictions(base_models, X)[0])

    for i, (name, z) in enumerate(meta_models_d.items()):
        predicted_error = z.predict(X)[0]

        # storing all meta_predictions
        step_results['all_meta_predictions'].append(predicted_error)
        # Storing all base_predictions
        step_results['all_base_predictions'].append(base_models[i].predict(X))

        # random bernoulli on confidence
        if meta_confidence_level:
            random_conf = np.random.binomial(1, z.get_confidence_score(), 1)
        else:
            random_conf = 1

        # random bernoulli inverse predicted error
        p = 1 - np.abs(predicted_error) / 100
        if p > 1 or p < 0:
            random_error = np.random.binomial(1, sys.float_info.epsilon, 1)
        else:
            random_error = np.random.binomial(1, p, 1)
        if random_conf * random_error:
            sub_meta_models[name] = z
            sub_base_models[name] = base_models[i]
            sub_meta_predictions[name] = predicted_error
            sub_base_predictions[name] = base_models[i].predict(X)[0]

    """Step of selection by diversity"""
    # Check if sub_base_models is empty
    if sub_base_models:
        # Seek n_best models
        heap = [(abs(value), key) for key, value in sub_meta_predictions.items()]
        smallest = heapq.nsmallest(n_best, heap)
        best_models_indexes = [key for value, key in smallest]

        # store idx of seed sub set of experts
        for idx in best_models_indexes:
            step_results['selected_experts_idx'].append(idx)

        # Initialize diversity dict
        sub_diverse_models = {key: sub_base_models[key] for key in best_models_indexes}
        sub_base_diverse_predictions = {key: sub_base_predictions[key] for key in best_models_indexes}
        n_best_models = list(sub_diverse_models.keys())

        #ranking approach
        if diversity_evaluator is not None and (diversity_measure == 'correlation' or diversity_measure == 'entropy'):
            potential_models = list(set(sub_base_models.keys()) - set(n_best_models))
            # TODO: check if not empty
            if len(potential_models) > 0:
                diversity_matrix = get_experts_diversity(diversity_evaluator, sub_base_models)
                index_to_add, index_to_remove = get_index(diversity_matrix, potential_models, n_best_models, threshold_div)

                for j in index_to_add:
                    sub_diverse_models[j] = sub_base_models[j]
                    sub_base_diverse_predictions[j] = sub_base_models[j].predict(X)[0]
                    step_results['selected_experts_idx'].append(j)
                for k in index_to_remove:
                    del sub_meta_models[k]
                    del sub_meta_predictions[k]
        #naive approach
        if diversity_measure == 'double_fault' or diversity_measure == 'disagree':
            # TODO: check base_m variable
            diversity_matrix = get_experts_diversity(diversity_evaluator, sub_base_models)
            for i in sorted(sub_diverse_models.keys()):
                index_to_add = get_div_index(diversity_matrix, diversity_measure, i, threshold_div)
                for j in index_to_add:
                    sub_diverse_models[j] = sub_base_models[j]
                    sub_base_diverse_predictions[j] = sub_base_models[j].predict(X)[0]
                    step_results['selected_experts_idx'].append(j)

    # convert dictionaries to lists
    final_sub_diverse_models = list(sub_diverse_models.values())
    final_sub_base_diverse_predictions = list(sub_base_diverse_predictions.values())

    final_sub_meta_models = [sub_meta_models[key] for key in sub_diverse_models.keys()]
    final_sub_meta_predictions = [sub_meta_predictions[key] for key in sub_diverse_models.keys()]
    return final_sub_meta_models, final_sub_diverse_models, final_sub_meta_predictions, final_sub_base_diverse_predictions, step_results


def get_all_predictions(base_models, X):
    """returns an array of shape (n_samples,n_models) of all models' predictions"""
    if X is not None:
        predictions = [[] for _ in range(len(base_models))]
        for i in range(len(base_models)):
            predictions[i].extend(base_models[i].predict(X))
        predictions = np.array(predictions).T
        return predictions


def get_experts_diversity(diversity_evaluator, sub_models=None):
    """Returns the correlation between models in sub_model_list only"""
    diversity_matrix = diversity_evaluator.get_diversity_matrix()
    models_to_keep = set(sub_models.keys())
    models_to_drop = set(diversity_matrix.columns) - models_to_keep
    if models_to_drop:
        experts_matrix = diversity_matrix.drop(index=models_to_drop, columns=models_to_drop)
        return experts_matrix
    else:
        return diversity_matrix


# Ranking potentiel models when using correlation
def score_potential_models(df, potential_models, n_best_models, threshold):
    score = {i: 0 for i in potential_models}
    for i in potential_models:
        for j in n_best_models:
            if df[i][j] <= threshold:
                score[i] += 1
            else:
                score[i] -= 1
    return score


def get_index(df, potential_models, n_best_models, threshold):
    index_to_remove = []
    index_to_add = []
    score = score_potential_models(df, potential_models, n_best_models, threshold)

    for i, s in score.items():
        if s == len(n_best_models):
            index_to_add.append(i)
        else:
            index_to_remove.append(i)

    return index_to_add, index_to_remove


def get_div_index(df, diversity_measure=None, i=0, threshold=None):
    if diversity_measure == 'disagree' or diversity_measure == 'dissimilarity':
        to_add = list(df.loc[df[i] >= threshold].index)
    elif diversity_measure == 'double_fault':
        to_add = list(df.loc[df[i] <= threshold].index)

    else:
        raise ValueError('diversity_measure must be')
    return to_add


def sequential_reweighting(selected_results_df, diversity_matrix):
    # rank according to predicted error ==> accuray
    selected_results_df = selected_results_df.sort_values(by=['meta'], ascending=True)

    # Compute initial weights for base-learners, weights will  be updated afterwards

    weights = weight_predictions(selected_results_df['meta'])[0]
    selected_results_df['weight'] = weights

    # Select best predicted base model
    # TODO : change this to select different selection methods
    selected_models_df = pd.DataFrame()
    selected_models_df = pd.concat([selected_models_df, selected_results_df.head(1)], ignore_index=True)

    # Remove selected model from selected_results_df
    selected_results_df = selected_results_df.iloc[1:]

    for index, model in selected_results_df.iterrows():
        # Iterate through all remaining models
        new_weight = model['weight']
        for idx, selected_model in selected_models_df.iterrows():
            pairwise_correlation = diversity_matrix.loc[model['id']][selected_model['id']]
            penalty_term = new_weight * selected_model['weight'] * pairwise_correlation

            selected_models_df.at[idx, 'weight'] = selected_model['weight'] + penalty_term
            model['weight'] -= penalty_term

        model_to_add_df = model.to_frame().T
        selected_models_df = pd.concat([selected_models_df, model_to_add_df], ignore_index=True)
    return selected_models_df


def select_threshold(df, competence_threshold):
    df['meta'] = np.abs(df['meta'])
    return df.loc[df['meta'] <= competence_threshold]


def select_percentage(df, selection_ratio):
    base_length = len(df.index)
    # Rank models according to meta predictions
    df = df.sort_values(by=['meta'], ascending=True)
    # TODO: check if ceil or int only
    return df.head(math.ceil(selection_ratio * base_length))


def select_nbest(df, n_best):
    df = df.sort_values(by=['meta'], ascending=True)

    return df.head(n_best)


def select_probability(df):
    random_variables = []
    # Normalize meta_predictions 
    df['meta'] = np.abs(df['meta'])
    meta_predictions = df['meta'].values.astype(float)
    min_meta = min(meta_predictions)
    max_meta = max(meta_predictions)
    
    df['meta'] = [(x-min_meta)/ (max_meta-min_meta) for x in df['meta']]

    for idx, row in df.iterrows():
        p = 1 - np.abs(row['meta'])
        """
        if p > 1 or p < 0:
            random_error = np.random.binomial(1, sys.float_info.epsilon, 1)
        else:"""
        random_error = np.random.binomial(1, p, 1)[0]

        random_variables.append(random_error)

    df['r'] = random_variables

    df = df.loc[df['r'] == 1]
    df = df.drop(columns=['r'])

    return df


def get_all_prediction_data_frame(X, meta_models, base_models):

    step_results = {'all_meta_predictions': [], 'all_base_predictions': [], 'selected_experts_idx': []}

    meta_predictions = []
    base_predictions = []

    # Getting all meta predictions
    for z in meta_models:
        meta = z.predict(X)
        meta_predictions.append(meta[0])
        step_results['all_meta_predictions'].append(meta[0])

    # getting all base_predictions
    for m in base_models:
        base = m.predict(X)
        base_predictions.append(base[0])
        step_results['all_base_predictions'].append(base[0])

    results_df = pd.DataFrame(
        {'id': [i for i in range(len(meta_models))], 'meta': meta_predictions, 'base': base_predictions})

    return results_df, step_results


def get_experts_prediction_abstaining(X, meta_models, base_models, selection_method, sequential_reweight, diversity_measure=None,
                                      competence_threshold=None, selection_ratio=None, n_best=None,
                                      diversity_evaluator=None):
    # TODO : check dimension of x
    results_df, step_results = get_all_prediction_data_frame(X=X, meta_models=meta_models, base_models=base_models)

    # Step 1 : select expert based on predicted error with  threshold or probability
    if selection_method == 'threshold':
        experts_df = select_threshold(results_df, competence_threshold)
    elif selection_method == 'probability':
        experts_df = select_probability(results_df)
    elif selection_method == 'percentage':
        experts_df = select_percentage(results_df, selection_ratio)
    elif selection_method == 'n_best':
        experts_df = select_nbest(results_df, n_best=n_best)

    selected_models_df = experts_df.copy(deep=True)
    estimation_method = None
    if not selected_models_df.empty:
        if sequential_reweight and not results_df.empty:
            # Compute diversity matrix
            if diversity_measure == 'redundancy':
                estimation_method = 'knn'
            diversity_matrix = diversity_evaluator.get_diversity_matrix(estimation_method)
            if diversity_measure == 'dissimilarity':
                diversity_matrix = 1-diversity_matrix

            selected_models_df = sequential_reweighting(results_df, diversity_matrix)
        else:
            weights = weight_predictions(selected_models_df['meta'])[0]
            selected_models_df['weight'] = weights

        step_results['selected_experts_idx'] = [int(i) for i in selected_models_df['id']]
    else:
        # Consider all base-models for prediction
        selected_models_df = results_df
        weights = weight_predictions(selected_models_df['meta'])[0]
        selected_models_df['weight'] = weights

    # Compute final prediction based on selected models only and recalculated weights
    base_predictions = np.array(selected_models_df['base'])
    final_prediction = get_aggregated_result(base_predictions, selected_models_df['weight'], method='weighted_average')
    return final_prediction, step_results

def select_experts(text):
    print(text)
