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
from os import listdir
from os.path import isfile, join



class ResultsComparator:

    @staticmethod
    def plot_comparison_results(files_dir_path, target_field, metric='mean_mse'):
        files = [f for f in listdir(files_dir_path) if isfile(join(files_dir_path, f)) if f.startswith(target_field+'_')]
        compare_df = pd.DataFrame()

        for f in files:
            df = pd.read_csv(join(files_dir_path, f), comment='#')
            compare_df['nb_instances'] = df['id']
            filter_col = [col for col in df if col.startswith(metric)]
            compare_df[f] = df[filter_col[0]]
        return compare_df

    @staticmethod
    def plot_comparison_results_size(files_dir_path, target_field, diversity='HETERO', abstain='VOTE', metric='global_mse'):
        files = [f for f in listdir(files_dir_path) if isfile(join(files_dir_path, f)) if
                 f.startswith(target_field + '_') and diversity in f and abstain in f]
        compare_df = pd.DataFrame()

        for f in files:
            df = pd.read_csv(join(files_dir_path, f), comment='#')
            compare_df['nb_instances'] = df['id']
            filter_col = [col for col in df if col.startswith(metric)]
            compare_df[f] = df[filter_col[0]]
        return compare_df

    @staticmethod
    def plot_comparison_results_type(files_dir_path, target_field, size, abstain='VOTE', metric='global_mse'):
        files = [f for f in listdir(files_dir_path) if isfile(join(files_dir_path, f)) if
                 f.startswith(target_field + '_') and str(size) in f and abstain in f]
        compare_df = pd.DataFrame()

        for f in files:
            df = pd.read_csv(join(files_dir_path, f), comment='#')
            compare_df['nb_instances'] = df['id']
            filter_col = [col for col in df if col.startswith(metric)]
            compare_df[f] = df[filter_col[0]]
        return compare_df

    @staticmethod
    def plot_htree_vs(files_dir_path, target_field, size, abstain, diversity, metric):
        files = [f for f in listdir(files_dir_path) if isfile(join(files_dir_path, f))
                 and f.startswith(target_field)
                 and str(size) in f
                 and abstain in f
                 and diversity in f
                 and str(size) in f]
        files.append([f for f in listdir(files_dir_path) if isfile(join(files_dir_path, f)) and
                  'HTREE' in f][0])
        compare_df = pd.DataFrame()

        for f in files:
            df = pd.read_csv(join(files_dir_path, f), comment='#')
            compare_df['nb_instances'] = df['id']
            filter_col = [col for col in df if col.startswith(metric)]
            compare_df[f] = df[filter_col[0]]
        return compare_df

    @staticmethod
    def plot_offline_vs(files_dir_path, target_field, size, abstain, diversity, metric):
        files = [f for f in listdir(files_dir_path) if isfile(join(files_dir_path, f))
                 and f.startswith(target_field + '_')
                 and str(size) in f
                 and abstain in f
                 and diversity in f
                 and '_' + str(size) in f]
        files.append([f for f in listdir(files_dir_path) if isfile(join(files_dir_path, f)) and
                      f.startswith(target_field + '_') and 'OFFLINE' in f][0])
        compare_df = pd.DataFrame()

        for f in files:
            df = pd.read_csv(join(files_dir_path, f), comment='#')
            compare_df['nb_instances'] = df['id']
            filter_col = [col for col in df if col.startswith(metric)]
            compare_df[f] = df[filter_col[0]]
        return compare_df



