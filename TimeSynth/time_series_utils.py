from os.path import join, splitext
from os.path import basename
from copy import deepcopy
from os import listdir
import os
import pandas as pd
from itertools import product


#from pymongo import MongoClient

MONGODB_ENDPOINT = 'mongodb://localhost:27017/'

LORA_DATA_COLLECTION_PREFIX = 'lora_data_'

mongo_client = MongoClient(MONGODB_ENDPOINT)


def read_collection_as_dataframe(db_name, collection_name):
     collection_df = pd.DataFrame(list(mongo_client[db_name][collection_name].find()))
     collection_df.sort_values(by=['tm'], ascending=True)
     return collection_df


def read_multiple_collection_as_dataframe(db_name, start_date, end_date):
    all_collections = mongo_client[db_name].collection_names()

    start_date_collection = LORA_DATA_COLLECTION_PREFIX + start_date
    end_date_collecition = LORA_DATA_COLLECTION_PREFIX + end_date
    selected_collections = [c for c in all_collections if start_date_collection < c < end_date_collecition]
    selected_collections.sort()

    final_df = pd.DataFrame()
    # Read collection of time interval dates
    for c in selected_collections:
        print(c)
        df = read_collection_as_dataframe(db_name, c)
        final_df = pd.concat([final_df, df], ignore_index=True, sort=False)

    # Sort ascending on tm
    final_df.sort_values(by=['tm'], ascending=True)
    return final_df


def filter_df_on_tp(df, tp_filter):
    df = df.loc[df['tp'] == tp_filter]
    return df


def count_nb_messages_by_interval(df, freq, date_field='tm', tp_filter=None):

    if tp_filter:
        df = filter_df_on_tp(df=df, tp_filter=tp_filter)

    df = df[['_id', date_field]]



    aggregatd_df = df.groupby(pd.Grouper(key=date_field, freq=freq)).count()  # or other function
    aggregatd_df = aggregatd_df.reset_index()
    aggregatd_df = aggregatd_df.rename(columns={'_id': 'target', 'tm': 'date'})
    return aggregatd_df


def count_nb_devices_by_interval(df, freq, date_field='tm', tp_filter=None):

    if tp_filter:
        df = filter_df_on_tp(df=df, tp_filter=tp_filter)

    df = df[['dvID', date_field]]

    aggregatd_df = df.groupby(pd.Grouper(key=date_field, freq=freq)).agg({'dvID': pd.Series.nunique}) # or other function
    # aggregatd_df = aggregatd_df.drop(columns=['tm'])
    aggregatd_df['tm'] = aggregatd_df.index
    aggregatd_df = aggregatd_df.reset_index(drop=True)
    aggregatd_df = aggregatd_df.rename(columns={'dvID': 'target', 'tm': 'date'})
    return aggregatd_df


def mean_signal_noise_ratio(df, freq, date_field='tm'):
    df = df[['sr', date_field]]
    # Signal noise to raition is in uplink only
    df = df.dropna()
    aggregatd_df = df.groupby(pd.Grouper(key=date_field, freq=freq)).mean()  # or other function
    aggregatd_df = aggregatd_df.reset_index()
    aggregatd_df = aggregatd_df.rename(columns={'sr': 'target', 'tm': 'date'})
    return aggregatd_df


def mean_packet_error(df, freq, date_field='tm', tp_filter=None):

    if tp_filter:
        df = filter_df_on_tp(df=df, tp_filter=tp_filter)
    df = df[['mPr', date_field]]
    # Signal noise to raition is in uplink only
    df = df.dropna()
    aggregatd_df = df.groupby(pd.Grouper(key=date_field, freq=freq)).mean()  # or other function
    aggregatd_df = aggregatd_df.reset_index()
    aggregatd_df = aggregatd_df.rename(columns={'mPr': 'target', 'tm': 'date'})
    return aggregatd_df


def store_dataframe_csv(df, output_file):
    df.to_csv(path_or_buf=output_file, index=False)


def plot_time_series(df, target):
    return None


def dateparse(x):
    try:
        return pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    except Exception:
        try:
            return pd.datetime.strptime(x, '%Y-%m-%d')
        except Exception as e:
            raise (e)


def create_lagged_time_series(file_path,  date_field, target_field, k_lags, header=None, df=None):
    if df is None:
        if header is not None:
            if date_field is not None: 
                ts_df = pd.read_csv(filepath_or_buffer=file_path, parse_dates=[date_field], date_parser=dateparse)
            else:
                ts_df = pd.read_csv(filepath_or_buffer=file_path)
        else:
            ts_df = pd.read_csv(filepath_or_buffer=file_path, header=header)
            
            ts_df.columns = [date_field, target_field]
            

    else:
        ts_df = deepcopy(df)
        
    if date_field is not None:
        ts_df.rename(columns={''+date_field: 'date', ''+target_field: 'target'}, inplace=True)
    else: 
        ts_df.rename(columns={target_field: 'target'}, inplace=True)
        
    print(ts_df.columns)

    target_field = 'target'
    for i in range(1, k_lags+1):
        ts_df['Tm{}'.format(str(i))] = ts_df[target_field].shift(i)
    ts_df = ts_df.dropna()
    return ts_df


def create_time_series_from_dir(src_dir, dst_dir, date_field='date', target_field='target', header=None, k_lags=7):
    for file_name in listdir(src_dir):
        ts_df = create_lagged_time_series(file_path=join(src_dir, file_name),
                                          date_field=date_field, target_field=target_field, header=header, k_lags=k_lags)
        ts_df.to_csv(path_or_buf=join(dst_dir, file_name), index=False)


def transform_multi_to_uni_variate(src_file, output_dir, date_field):
    mv_df = pd.read_csv(filepath_or_buffer=src_file,parse_dates=[date_field], date_parser=dateparse)
    file_name = basename(src_file)
    file_name, extension = splitext(file_name)

    for c in mv_df.columns:
        if c != date_field:
            uv_df = mv_df[[date_field, c]]
            output_file_name = file_name + '_' + str(c) + extension
            uv_df = create_lagged_time_series(df=uv_df, date_field=date_field, k_lags=7, target_field=str(c), file_path=None)
            uv_df.to_csv(path_or_buf=join(output_dir, output_file_name), index=False)


if __name__ == '__main__':
    base_dir = '/media/dihia/DATADRIVE11/Thesis/code/streaming_ade/data/datasets/0-fake_data'
    src_dir = join(base_dir, 'test_not_lagged')
    dst_dir = join(base_dir, 'test')
    create_time_series_from_dir(src_dir=join(base_dir,src_dir),
                                dst_dir=join(base_dir,dst_dir), k_lags=7, header=True, target_field='target', date_field= None)

    '''
    data_base_name = 'lora_data'
    start_date = '2019-05-18'
    end_date = '2019-05-28'

    fields = ['messages', 'devices', 'mean_sr', 'mean_per']
    time_intervals = [1, 5, 10, 15, 30, 60]
    tp_filters = [None, 0, 1, 2]
    OUTPUT_DIR = '/media/dihia/DATADRIVE1/Thesis/code/streaming-ade/data/'

    data_set_dir = '_'.join(['lora', start_date, end_date])
    data_output_dir = join(OUTPUT_DIR, data_set_dir)

    if not os.path.exists(data_output_dir):
        os.makedirs(data_output_dir)

    # aggregated_df = count_nb_messages_by_interval(df=df, freq='60Min')

    # Get all data


    # all_data_df = read_multiple_collection_as_dataframe(db_name=data_base_name, start_date=start_date, end_date=end_date)

    all_collections = mongo_client[data_base_name].collection_names()

    start_date_collection = LORA_DATA_COLLECTION_PREFIX + start_date
    end_date_collecition = LORA_DATA_COLLECTION_PREFIX + end_date
    selected_collections = [c for c in all_collections if start_date_collection < c < end_date_collecition]
    selected_collections.sort()

    for c in selected_collections:
        print(c)
        all_data_df = read_collection_as_dataframe(db_name=data_base_name, collection_name=c)

        for field, time_interval, tp_filter in product(fields, time_intervals, tp_filters):
            freq = str(time_interval) + 'Min'

            if field == 'messages':
                # Count nb_messages by time interval

                df = count_nb_messages_by_interval(df=all_data_df, freq=freq, tp_filter=tp_filter)
            elif field == 'devices':
                # Count number of unique devices by time interval

                df = count_nb_devices_by_interval(df=all_data_df, freq=freq, tp_filter=tp_filter)
            elif field == 'mean_sr':
                # Mean Signa Noise Ratio in the time interval
                df = mean_signal_noise_ratio(df=all_data_df, freq=freq)
            elif field =='mean_per':
                # Mean Packet Error in the time interval
                df = mean_packet_error(df=all_data_df, freq=freq, tp_filter=tp_filter)
            else:
                raise ValueError('Not valid field values', field)

            sub_dir = join(data_output_dir, field)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            file_name_parts = [data_set_dir, field, 'interval', str(time_interval)]
            if tp_filter:
                file_name_parts.append('tp_' + str(tp_filter))

            file_name = '_'.join(file_name_parts) + '.csv'
            file_path = join(sub_dir, file_name)

            if not os.path.exists(file_path):
                df.to_csv(path_or_buf=file_path, index=False)
            else:
                # Write concat dataframes
                with open(file_path, 'a') as f:
                    # df.to_csv(f, header=False)
                    df.to_csv(f, index=False, header=False)

            # TODO: store aggregated df to csv'''

