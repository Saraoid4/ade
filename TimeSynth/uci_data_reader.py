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


class UCIReader:
    def __init__(self):
        self.data_frame = None

    def read_uci_url(self, url, header=None, names=None, date_columns = None):
        self.data_frame = pd.read_csv(url, header=header, names=names, parse_dates=date_columns, na_values='?').dropna()

    def write_repo_to_csv(self, output_file, header=None, sep=','):
        self.data_frame.to_csv(output_file, sep=sep, index=False)

    def drop_columns(self, columns):
        self.data_frame = self.data_frame.drop(columns, axis=1)

    def flatten_date_field(self,date_field):

        date_serie = self.data_frame[date_field]
        self.data_frame['Year'] = [c.year for c in date_serie]
        self.data_frame['Month'] = [c.month for c in date_serie]
        self.data_frame['Day'] = [c.day for c in date_serie]

    def flatten_time_field(self):
        raise NotImplementedError
