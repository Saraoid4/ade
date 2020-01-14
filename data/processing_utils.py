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
from os.path import join


def flatten_date_field(data_frame, date_field):
    date_serie = data_frame[date_field]
    data_frame['Year'] = [c.year for c in date_serie]
    data_frame['Month'] = [c.month for c in date_serie]
    data_frame['Day'] = [c.day for c in date_serie]


def flatten_time_field(data_frame, time_field):
    time_serie = data_frame[time_field]
    data_frame['Hour'] = [c.hour for c in time_serie]
    data_frame['Minute'] = [c.minute for c in time_serie]
    data_frame['Second'] = [c.second for c in time_serie]


def drop_nan_columns(data_frame):
    df = data_frame.dropna(axis=1, how='all')
    return df


def drop_columns(data_frame, columns_to_drop):
    return data_frame.drop(columns=columns_to_drop, axis=1)


def prepare_data_for_use(data_name, csv_file_path, output_dir_path, date_field='date', sep=',', add_time_fields=True,
                         id_field=False):

    def dateparse(x):
        try:
            return pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        except Exception:
            try:
                return pd.datetime.strptime(x, '%Y-%m-%d')
            except Exception as e:
                raise(e)
    if date_field:
        df = pd.read_csv(csv_file_path, sep=sep, parse_dates=[date_field], date_parser=dateparse)
    else :
        df = pd.read_csv(csv_file_path)
    if id_field:
        df = df.drop(columns=[id_field])

    if add_time_fields:
        df['year'] = [c.year for c in df[date_field]]
        df['month'] = [c.month for c in df[date_field]]
        df['day'] = [c.day for c in df[date_field]]
        df['hour'] = [c.hour for c in df[date_field]]
        df['minute'] = [c.minute for c in df[date_field]]
        df['second'] = [c.second for c in df[date_field]]

    if add_time_fields :
        df = df.drop(columns=[date_field])
    df.to_csv(join(output_dir_path, data_name+'.csv'),sep=sep, index=False)




