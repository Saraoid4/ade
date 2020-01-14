# !/usr/bin/env python
__author__ = "Dihia BOULEGANE"
__copyright__ = ""
__credits__ = ["Dihia BOULEGANE"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Dihia BOULEGANE"
__email__ = "dihia.boulegane@telecom-paristech.fr"
__status__ = "Development"

import os
import random
import pandas as pd
from os import listdir
from copy import deepcopy
from itertools import product, repeat
from multiprocessing import Process, Pool
from ade.ade_factory import ArbitratedEnsembleFactory
from skmultiflow.trees import RegressionHAT, RegressionHoeffdingTree
from skmultiflow.lazy.knn_forecast import KNNForecast
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.core import Pipeline
from skmultiflow.data import FileStream
from data.processing_utils import *
from core.meta_model import MetaModel
from skmultiflow.data import RegressionGenerator

# Setting up constants
OUTPUT_DIR = 'outputs/'
DATASET_DIR = 'data/datasets/'
DIVERSITY_OUTPUT_DIR = 'diversity_outputs/'

# Base-models parameters values
BASE_MODELS = ['KNN', 'KNN_W', "HTREE", "HATREE"]
NB_UNIQUE_BASE_MODELS = len(BASE_MODELS)
NB_NEIGHBORS_VALUES = list(range(2, 10))
LEAF_PREDICTION_VALUES = ['mean', 'perceptron']
MAX_WIDOW_SIZE = list(range(100, 2000, 100))

PARAMETERS_DICT = {
	'n_neighbors': NB_NEIGHBORS_VALUES,
	'leaf_prediction': LEAF_PREDICTION_VALUES,
	'max_window_size': MAX_WIDOW_SIZE
}

# ADE parameters settings section
ABSTAIN_VALUES = [False]
ABSTAIN_TWICE_VALUES = [False]
THRESHOLD_VALUES = [False]

PROBABILITY_VALUES = [False]

DIVERSITY_VALUES = [False]


NAIVE_VALUES = [False]

PERC_SELECT = [False]

RE_WEIGHT = [False]

N_BEST_VALUES = [False]
N_BEST_PRAMS = {'n_best': [1]}

# ADE parameters values section
# Update : Meta confidence level is not in the basic params, only related to threshold and probability
META_CONFIDENCE_LEVEL_VALUES = [False, True]
META_CONFIDENCE_PARAMS = {'meta_confidence_level': META_CONFIDENCE_LEVEL_VALUES}

# Global ADE parameters
META_ERROR_METRIC_VALUES = ['MAPE']

# All basic parameters should be included here
BASIC_ADE_PARAMS = {'meta_error_metric': META_ERROR_METRIC_VALUES}

# Threshold-based parameters values
THRESHOLD_METHOD_VALUES = ['product', 'sum', 'static']
# THRESHOLD_METHOD_VALUES = ['static']
THRESHOLD_UPDATE_STEP_VALUES = [0.01]
COMPETENCE_THRESHOLD_VALUES = [20]
# All threshold-based related parameters should be included here
THRESHOLD_ADE_PARAMS = {'competence_threshold': COMPETENCE_THRESHOLD_VALUES,
                        'threshold_update_method': THRESHOLD_METHOD_VALUES,
                        'threshold_update_step': THRESHOLD_UPDATE_STEP_VALUES}

# Diversity parameters values
DIVERSITY_METHOD_VALUES = ['sliding_window', 'fading factor', 'incremental']
DIVERSITY_MEASURE_VALUES = ['correlation', 'disagree', 'double_fault']

DIVERSITY_THRESHOLD_VALUES = [0.75]
N_SLIDING_VALUES = [200]
FADING_FACTOR_VALUES = [0.995]
# All diversity parameters should be included here
DIVERSITY_ADE_PARAMS = {'diversity_method': DIVERSITY_METHOD_VALUES,
                        'diversity_measure': DIVERSITY_MEASURE_VALUES,
                        'diversity_threshold': DIVERSITY_THRESHOLD_VALUES,
                        'n_sliding': N_SLIDING_VALUES,
                        'fading_factor': FADING_FACTOR_VALUES,
                        }

# MMR parameters values
MMR_CRITERION = [False]
MMR_THRESHOLD_VALUES = [0.5]
# MMR_TRADE_OFF_MEASURES = ['relevance_redundancy', 'accu_div']
MMR_TRADE_OFF_MEASURES = ['accu_div']
TRADE_OFF_LAMBDA = [0.7]

MMR_ADE_PARAMS = {'mmr_threshold': MMR_THRESHOLD_VALUES,
                  'trade_off': MMR_TRADE_OFF_MEASURES,
                  'trade_off_lambda': TRADE_OFF_LAMBDA,
                  'n_sliding': N_SLIDING_VALUES}

# Sequential reweighting values
SEQ_REWEIGHT_VALUES = [False]
SEQ_REWEIGHT_PARAMS = {'sequential_reweight': RE_WEIGHT}

# Not used settings
OFFLINE_VALUES = [False]
HETERO_VALUES = [True]
HUGE_VALUES = [True]

ade_factory = ArbitratedEnsembleFactory()


def _get_ade_settings(**settings_values):
	# TODO: setp up values for sequential weight and percentage selection and MMR

	abstain_values = settings_values['abstain']
	abstain_twice_values = settings_values['abstain_twice']
	threshold_selection_values = settings_values['threshold_selection']
	probability_selection_values = settings_values['probability_selection']
	diversity_selection_values = settings_values['diversity_selection']
	# Setting up values new selection methods
	mmr_criterion_selection_values = settings_values['mmr_criterion_selection']
	percentage_selection_values = settings_values['percentage_selection']
	naive_values = settings_values['naive']
	n_best_values = settings_values['n_best_selection']

	global_params = []
	for abstain, abstain_twice, threshold_selection, probability_selection, diversity_selection, mmr_criterion_selection, \
	    percentage_selection, naive, n_best_selection in product(abstain_values, abstain_twice_values,
	                                                             threshold_selection_values,
	                                                             probability_selection_values,
	                                                             diversity_selection_values,
	                                                             mmr_criterion_selection_values,
	                                                             percentage_selection_values,
	                                                             naive_values,
	                                                             n_best_values):

		params_dict = {'abstain': abstain,
		               'abstain_twice': abstain_twice,
		               'threshold_selection': threshold_selection,
		               'probability_selection': probability_selection,
		               'diversity_selection': diversity_selection,
		               'mmr_criterion_selection': mmr_criterion_selection,
		               'percentage_selection': percentage_selection,
		               'naive': naive,
		               'n_best_selection': n_best_selection,
		               'args_dict': {}}

		if sum([abstain, abstain_twice]) == 1:
			if abstain:
				# Expect one and only one from threshold or prob or sequential or percetange or MMR to be True
				if sum([threshold_selection, probability_selection, percentage_selection,
				        mmr_criterion_selection, n_best_selection]) == 1 and not naive:

					if mmr_criterion_selection and not diversity_selection:
						# Adding MMR paramaters
						mmr_param_names = list(MMR_ADE_PARAMS.keys())
						for mmr_param_values in product(*MMR_ADE_PARAMS.values()):
							args_dict = deepcopy(dict(zip(mmr_param_names, mmr_param_values)))
							params_dict['args_dict'].update(args_dict)
							global_params.append(deepcopy(params_dict))

					# TODO: Clean code from update parameters repetition
					elif threshold_selection:
						# Adding threshold parameters
						th_param_names = list(THRESHOLD_ADE_PARAMS.keys())
						for th_param_values in product(*THRESHOLD_ADE_PARAMS.values()):
							args_dict = deepcopy(dict(zip(th_param_names, th_param_values)))
							params_dict['args_dict'].update(args_dict)
							if params_dict['diversity_selection']:
								div_param_names = list(DIVERSITY_ADE_PARAMS.keys())
								for div_param_values in product(*DIVERSITY_ADE_PARAMS.values()):
									div_args_dict = deepcopy(dict(zip(div_param_names, div_param_values)))
									params_dict['args_dict'].update(div_args_dict)
									global_params.append(deepcopy(params_dict))
							else:
								w_param_names = list(SEQ_REWEIGHT_PARAMS.keys())
								for w_param_values in product(*SEQ_REWEIGHT_PARAMS.values()):
									w_args_dict = deepcopy(dict(zip(w_param_names, w_param_values)))
									params_dict['args_dict'].update(w_args_dict)
									global_params.append(deepcopy(params_dict))

					elif probability_selection:
						# TODO : clean after
						if params_dict['diversity_selection']:
							div_param_names = list(DIVERSITY_ADE_PARAMS.keys())

							for div_param_values in product(*DIVERSITY_ADE_PARAMS.values()):
								div_args_dict = deepcopy(dict(zip(div_param_names, div_param_values)))
								params_dict['args_dict'].update(div_args_dict)
								# add meta-conf params
								meta_conf_params = list(META_CONFIDENCE_PARAMS.keys())
								for meta_conf_values in product(*META_CONFIDENCE_PARAMS.values()):
									meta_args_dict = deepcopy(dict(zip(meta_conf_params, meta_conf_values)))
									params_dict['args_dict'].update(meta_args_dict)
									global_params.append(deepcopy(params_dict))

						else:
							w_param_names = list(SEQ_REWEIGHT_PARAMS.keys())
							for w_param_values in product(*SEQ_REWEIGHT_PARAMS.values()):
								w_args_dict = deepcopy(dict(zip(w_param_names, w_param_values)))
								params_dict['args_dict'].update(w_args_dict)
								# add meta-conf params
								meta_conf_params = list(META_CONFIDENCE_PARAMS.keys())
								for meta_conf_values in product(*META_CONFIDENCE_PARAMS.values()):
									meta_args_dict = deepcopy(dict(zip(meta_conf_params, meta_conf_values)))
									params_dict['args_dict'].update(meta_args_dict)
									global_params.append(deepcopy(params_dict))

					elif percentage_selection and not diversity_selection:
						# TODO: change percentage to parameter
						w_param_names = list(SEQ_REWEIGHT_PARAMS.keys())
						for w_param_values in product(*SEQ_REWEIGHT_PARAMS.values()):
							w_args_dict = deepcopy(dict(zip(w_param_names, w_param_values)))
							params_dict['args_dict'].update(w_args_dict)
							global_params.append(deepcopy(params_dict))
					# Return Nbest ADE
					elif n_best_selection:
						best_params_names = list(N_BEST_PRAMS.keys())
						for best_param_values in product(*N_BEST_PRAMS.values()):
							best_args_dict = deepcopy(dict(zip(best_params_names, best_param_values)))
							params_dict['args_dict'].update(best_args_dict)
							global_params.append(deepcopy(params_dict))

			if abstain_twice and not any([diversity_selection, mmr_criterion_selection,
			                              percentage_selection, naive]):

				if threshold_selection and not probability_selection:
					# Adding threshold parameters
					th_param_names = list(THRESHOLD_ADE_PARAMS.keys())
					for th_param_values in product(*THRESHOLD_ADE_PARAMS.values()):
						args_dict = deepcopy(dict(zip(th_param_names, th_param_values)))
						params_dict['args_dict'].update(args_dict)
						global_params.append(deepcopy(params_dict))
				if probability_selection and not threshold_selection:
					# add meta-conf params
					meta_conf_params = list(META_CONFIDENCE_PARAMS.keys())
					for meta_conf_values in product(*META_CONFIDENCE_PARAMS.values()):
						meta_args_dict = deepcopy(dict(zip(meta_conf_params, meta_conf_values)))
						params_dict['args_dict'].update(meta_args_dict)
						global_params.append(deepcopy(params_dict))
		else:
			# Expect Naive or Weighted
			if not any([abstain, abstain_twice, threshold_selection, probability_selection,
			            percentage_selection, mmr_criterion_selection, diversity_selection]):
				# return a simple ensemble or a weighted ensemble
				params_dict = {'abstain': abstain,
				               'abstain_twice': abstain_twice,
				               'threshold_selection': threshold_selection,
				               'probability_selection': probability_selection,
				               'diversity_selection': diversity_selection,
				               'mmr_criterion_selection': mmr_criterion_selection,
				               'percentage_selection': percentage_selection,
				               'naive': naive,
				               'n_best_selection': n_best_selection,
				               'args_dict': {}}
				global_params.append(deepcopy(params_dict))
	# Adding basic parameters
	final_parameters_dict_list = []
	final_basic_params = []
	basic_param_names = list(BASIC_ADE_PARAMS.keys())
	for basic_param_values in product(*BASIC_ADE_PARAMS.values()):
		basic_args_dict = deepcopy(dict(zip(basic_param_names, basic_param_values)))
		final_basic_params.append(basic_args_dict)

	if global_params:
		for basic_item, global_item in product(final_basic_params, global_params):
			temp_global = deepcopy(global_item)
			temp_basic = deepcopy(basic_item)
			temp_global['args_dict'].update(temp_basic)
			final_parameters_dict_list.append(deepcopy(temp_global))
	else:

		for basic_args_dict in final_basic_params:
			final_parameters_dict_list.append({'abstain': abstain, 'abstain_twice': abstain_twice,
			                                   'threshold_selection': threshold_selection,
			                                   'probability_selection': probability_selection,
			                                   'diversity_selection': diversity_selection,
			                                   'mmr_criterion_selection': mmr_criterion_selection,
			                                   'percentage_selection': percentage_selection,
			                                   'n_best_selection': n_best_selection,
			                                   'args_dict': deepcopy(basic_args_dict)})
	return final_parameters_dict_list


def _get_base_model(base_model_name, model_parameters=None):
	if base_model_name == "KNN":
		model = KNNForecast()
	elif base_model_name == 'KNN_W':
		model = KNNForecast(weighted=True)
	elif base_model_name == 'HTREE':
		model = RegressionHoeffdingTree()

	elif base_model_name == "HATREE":
		model = RegressionHAT()
	else:
		raise ValueError(base_model_name, ' Unknown base model name please check that value is in', BASE_MODELS)

	if model_parameters is not None:
		# Setup up parameters for
		for k, v in model_parameters.items():
			if k in model.__dict__.keys():
				setattr(model, k, v)
	return model


def _get_meta_model_ensemble(model_type, ensemble_size):
	meta_ensemble = []
	for i in range(ensemble_size):
		meta_ensemble.append(MetaModel(model_type=model_type, confidence_fading_factor=0.995))
	return meta_ensemble


def init_test_models(ensemble_size, data_name):
	models = {}

	settings_values = {'abstain': ABSTAIN_VALUES, 'abstain_twice': ABSTAIN_TWICE_VALUES,
	                   'threshold_selection': THRESHOLD_VALUES,
	                   'probability_selection': PROBABILITY_VALUES, 'diversity_selection': DIVERSITY_VALUES,
	                   'mmr_criterion_selection': MMR_CRITERION,
	                   'percentage_selection': PERC_SELECT,
	                   'naive': NAIVE_VALUES,
	                   'n_best_selection': N_BEST_VALUES
	                   }

	# Get different ADE settings
	test_settings = _get_ade_settings(**settings_values)
	# Init base and meta models
	base_ensemble = _init_huge_diverse_ensembles(parameters=PARAMETERS_DICT, ensemble_size=ensemble_size)
	meta_ensemble = _get_meta_model_ensemble(model_type='HATREE', ensemble_size=len(base_ensemble))

	for data in test_settings:
		# TODO: add factory  and get ensemble here

		data['args_dict'].update({'meta_models': deepcopy(meta_ensemble), 'base_models': deepcopy(base_ensemble)})
		# Set up ensemble output file

		try:
			data['args_dict'].update({'output_file': join(DIVERSITY_OUTPUT_DIR, data_name)})
			ensemble = ade_factory.get_ensemble(**data)
			ensemble_tag = ensemble.get_model_name()
			models[ensemble_tag] = Pipeline([('Classifier', ensemble)])
		except Exception as e:
			print(e)
			pass

	return models


def _select_random_models(base_ensemble, size):
	random.shuffle(base_ensemble)
	return random.sample(base_ensemble, size)


def _init_huge_diverse_ensembles(parameters, ensemble_size, model_name=None):
	parameters_keys = list(parameters.keys())
	models = []

	if not model_name:
		model_names = BASE_MODELS
	else:
		model_names = [model_name]

	for model_name in model_names:

		for l in product(*parameters.values()):
			parameters_dict = {}
			for i in range(len(parameters_keys)):
				parameters_dict[parameters_keys[i]] = l[i]

			model = _get_base_model(model_name, parameters_dict)
			models.append(model)

	base_ensemble = _select_random_models(models, ensemble_size)
	return base_ensemble


def create_dataset_list(dataset_dir, date=True, competence_threshold=1, id=False):
	datasets = []
	for f in listdir(dataset_dir):
		name = f.replace('.csv', '')
		dataset = {'name': name, 'date': date, 'competence_threshold': competence_threshold, 'id': id}
		datasets.append(dataset)

	return datasets


def evaluate_ensemble(key_model, value_model, dataset_dirname, data_output_dir, data_name):
	data_file = data_name + '.csv'
	stream_csv_file = join(DATASET_DIR, dataset_dirname, TEMP_DATA_DIR, data_file)
	# TODO: better way to get target idx
	df_header = pd.read_csv(stream_csv_file, nrows=1)
	df_header = list(df_header.columns)
	target_idx = df_header.index('target')

	stream = FileStream(stream_csv_file, target_idx=target_idx, n_targets=1)
	stream.prepare_for_use()

	# stream = RegressionGenerator(n_samples=100, n_features=2)
	# print("     ", key_model, "\n")
	print(key_model, data_name)

	output_file_path = join(data_output_dir, data_name + '_' + key_model + '.csv')

	evaluator = EvaluatePrequential(pretrain_size=10, max_samples=3000000, batch_size=1, n_wait=1,
	                                max_time=10000000,
	                                output_file=output_file_path, show_plot=False,
	                                data_points_for_classification=False, restart_stream=True,
	                                metrics=['mean_square_error', 'true_vs_predicted'])

	model_name = data_name + '_' + key_model
	models = [value_model]
	model_names = [model_name]

	evaluator.evaluate(stream=stream, model=models, model_names=model_names)


if __name__ == '__main__':

	# TODO: improve ensemble size steps and ranges
	ensemble_size_range = [10]
	TEST_DATA_DIR = 'test'
	TEMP_DATA_DIR = 'temp'
	OUTPUT_DIR = 'outputs'
	date_field = None
	sep = ','

	competence_threshold = 20
	dirs = listdir(DATASET_DIR)[-1:]
	dirs.sort()

	global_models_keys = []
	global_models_values = []
	global_dataset_dir_name = []
	global_data_output_dir = []
	global_data_name = []

	# Pool of parallel processes
	processes = Pool(processes=5)

	for dataset_dirname in dirs[0:1]:

		dataset_path = os.path.join(DATASET_DIR, dataset_dirname, TEST_DATA_DIR)
		# TODO: correct this later for more re-usability

		if 'real' in dataset_dirname or 'lora' in dataset_dirname or 'bitcoin' in dataset_dirname:

			datasets = create_dataset_list(dataset_path, date=False, competence_threshold=competence_threshold, id=False)
		else:
			raise ValueError('Wrong Dataset Direcotry')

		for ensemble_size in ensemble_size_range:

			for dataset in datasets[0:1]:

				print("\n ######## Processing:", dataset, "\n")
				data_name = dataset['name']
				competence_threshold = dataset['competence_threshold']

				models = init_test_models(ensemble_size, data_name)
				print(models)
				data_file = data_name + '.csv'

				csv_file_path = join(DATASET_DIR, dataset_dirname, TEST_DATA_DIR, data_file)
				if dataset['date']:
					prepare_data_for_use(data_name, csv_file_path, join(DATASET_DIR, dataset_dirname, TEMP_DATA_DIR))

				elif dataset['id']:
					prepare_data_for_use(data_name, csv_file_path, TEMP_DATA_DIR, date_field=None,
					                     add_time_fields=False,
					                     id_field='id')

				data_output_dir = join(OUTPUT_DIR, dataset_dirname, data_name)
				if not os.path.exists(data_output_dir):
					os.makedirs(data_output_dir)

				# Parallel function here
				for key, value in models.items():
					global_models_keys.append(key)
					global_models_values.append(value)
					global_dataset_dir_name.append(dataset_dirname)
					global_data_output_dir.append(data_output_dir)
					global_data_name.append(data_name)

	# print(len(global_models_keys))
	processes.starmap_async(evaluate_ensemble, zip(global_models_keys, global_models_values, global_dataset_dir_name,
	                                               global_data_output_dir, global_data_name))


