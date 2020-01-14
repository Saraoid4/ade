import os
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt

from datetime import datetime
from os.path import join


plt.style.use('fivethirtyeight')


def read_btc_data(currency, start_date, end_date):
    btc = web.get_data_yahoo(currency, start=start_date, end=end_date)
    return btc


def datetime_to_string(dt, string_format='%Y-%m-%d'):
    return dt.strftime(string_format)

def get_btc_data(currency, start_date, end_date):
    start_date = datetime(2018, 1, 1)
    end_date = datetime.now()
    btc_df = read_btc_data(currency='BTC-USD', start_date=start_date, end_date=end_date)

    # Saving data to csv file
    # Get current direcotry
    cwd = os.getcwd()
    btc_df.to_csv(join(cwd, 'btc_{0}_{1}.csv'.format(datetime_to_string(start_date), datetime_to_string(end_date))))



def read_btc_data_csv(file_path, header=None):
    btc_df = pd.read_csv(file_path, header=header)
    btc_df.columns = ['date', 'target']
    btc_df.to_csv(file_path)








