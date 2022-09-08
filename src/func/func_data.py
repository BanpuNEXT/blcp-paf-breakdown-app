import numpy as np
import pandas as pd
import datetime as dt
from bson.objectid import ObjectId
from sklearn.preprocessing import StandardScaler
import joblib
from flask import abort


def get_device_name(device_id):
    '''
    Get device_name from device_id to load model and configuration value.
    '''
    device_dict = {
        '62ff07274adcc50b51965f96': 'PAF_1A',
        '62ff07274adcc50b51965f97': 'PAF_2A',
        '62ff07274adcc50b51965f98': 'PAF_1B',
        '62ff07274adcc50b51965f99': 'PAF_2B'
    }

    try:
        device_name = device_dict[device_id]
    except KeyError:
        abort(400, description=f"deviceId not found.")

    return device_name


def get_device_id(device_name):
    '''
    Get ObjectId device_id from device_name to GET data from database.
    '''
    device_dict = {
        'PAF_1A': ObjectId('62ff07274adcc50b51965f96'),
        'PAF_2A': ObjectId('62ff07274adcc50b51965f97'),
        'PAF_1B': ObjectId('62ff07274adcc50b51965f98'),
        'PAF_2B': ObjectId('62ff07274adcc50b51965f99')
    }

    device_id = device_dict[device_name]

    return device_id


def get_training_date(start_training_date, n_training_day, n_test_day):
    '''
    Get training date from config file and transform to dt_datetime.
    '''
    start_date = dt.datetime.strptime(start_training_date, '%Y-%m-%d %H:%M:%S')
    end_date = start_date + dt.timedelta(days=n_training_day) + dt.timedelta(days=n_test_day)

    return start_date, end_date


def get_predict_date(db):
    '''
    Get latest 90 days data to predict.
    '''
    end_date = db['data_raw'].find_one(sort=[('publishTimestamp', -1)])['publishTimestamp']
    start_date = end_date - dt.timedelta(days=90)

    return start_date, end_date


def get_query(device_id, start_date, end_date):
    '''
    Create mongodb query to GET data.
    '''
    query = {
        'deviceId': device_id,
        'publishTimestamp': {
            '$gte': start_date,
            '$lte': end_date
        }
    }

    return query


def load_data(query, db):
    '''
    GET data from mongodb.
    '''
    cursor = db['data_raw'].find(query, {
        '_id': 0,
        'publishTimestamp': 1,
        'generator_mw': 1,
        'vibration_nde_um': 1,
        'vibration_de_um': 1,
        'temp_nde_degc': 1,
        'temp_de_degc': 1,
        'amb_temp_degc': 1
        })

    data_list = list(cursor)
    df = pd.json_normalize(data_list)
    df = df.sort_values(by=['publishTimestamp'])
    df.index = df['publishTimestamp']
    df = df.drop(columns=['publishTimestamp'])
    
    return df


def clean_type(df):
    '''
    Transfrom all data to float and replace all string to NaN.
    '''
    df = df.replace('Calc Failed|Bad|Under Range|Comm Fail', np.nan, regex=True)
    dtype_dict = {
        'generator_mw': float,
        'vibration_nde_um': float,
        'vibration_de_um': float,
        'temp_nde_degc': float,
        'temp_de_degc': float,
        'amb_temp_degc': float
    }
    
    try:
        df = df.astype(dtype_dict)
    except ValueError:
        abort(500, description=f"String input found in database.")

    return df


def add_additional_feature(df):
    '''
    Add diff_temp_degc from device temp - ambient temp.
    '''
    df['diff_nde_temp_degc'] = df['temp_nde_degc'] - df['amb_temp_degc']
    df['diff_de_temp_degc'] = df['temp_de_degc'] - df['amb_temp_degc']
    
    df = df.drop(columns=[
        'amb_temp_degc',
        'temp_nde_degc',
        'temp_de_degc'
        ])

    return df


def filter_data(df, low_generation_threshold):
    '''
    Transfrom low generation records to NaN to exclude from prediction.
    '''
    df.loc[df['generator_mw'] <= low_generation_threshold, :] = np.nan
    
    return df


def filter_feature(df, feature):
    '''
    Select feature for single variable models.
    '''
    try:
        df = df[[feature]]
    except KeyError:
        abort(400, description=f"feature not found.")

    return df


def split_data(df, n_training_day, n_look_back_day):
    '''
    Divide first n_training_day from model configuration file to training set.
    Divide the rest to test set.
    '''
    split_date = df.index[0] + dt.timedelta(days=n_training_day)
    training_df = df[df.index < split_date]
    test_df = df[df.index >= split_date - dt.timedelta(days=n_look_back_day)]
    
    return training_df, test_df, split_date


def transform_scale(training_df, test_df, scaler_path, device_name, feature):
    '''
    Fit scaler to training set and transform to test set.
    '''
    sc = StandardScaler()
    training_df[[feature]] = sc.fit_transform(training_df[[feature]])
    test_df[[feature]] = sc.transform(test_df[[feature]])
    
    joblib.dump(sc, f'{scaler_path}/{device_name}_{feature}.save') 
    
    return training_df, test_df


def load_scale(input_df, scaler_path, device_name, feature):
    '''
    Transform predict set by scaler from training set.
    '''
    try:
        sc = joblib.load(f'{scaler_path}/{device_name}_{feature}.save')
    except FileNotFoundError:
        abort(422, description=f"Not available scaler for {device_name}: {feature}.")
    
    input_df[[feature]] = sc.transform(input_df[[feature]])

    return input_df


def divide_non_operation(df, minute_interval):
    '''
    Assign group for sequence of normal generation for creating array of X.
    '''
    group_df = df.copy()
    group_df['publishTimestamp'] = group_df.index
    group_df = group_df.dropna()
    group_df['diff_time'] = group_df['publishTimestamp'].diff()
    group_df.loc[group_df.index[0], 'diff_time'] = pd.Timedelta(f'{minute_interval}m')
    group_df.loc[group_df.index[0:], 'diff_time'] = group_df.loc[group_df.index[1:], 'diff_time'].apply(lambda x: int(x.total_seconds() / 60))
    group_df['check_sequence'] = group_df['diff_time'].apply(lambda x: 0 if x == minute_interval else 1)
    group_df['group'] = group_df['check_sequence'].cumsum()

    group_df.drop(columns=['diff_time'])

    return group_df


def create_sequences(values, time_steps):
    '''
    A function to create matrix format for model input.
    '''
    output = []

    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    
    return np.stack(output), np.stack(output)


def cal_time_steps(minute_interval, n_look_back_day):
    '''
    Calculate length of time dimension of an array in the input matrix.
    '''
    try:
        time_steps = int((60 / minute_interval) * 24) * n_look_back_day
    except ZeroDivisionError:
        abort(422, description=f"minute_interval in configuration file must not be 0.")

    return time_steps


def group_sequence(df, feature, time_steps):
    '''
    Put divided sequence data into matrix format for model input.
    '''
    valid_timestamp = []

    # Create padding zeros to be concantinated with looping result.
    x = np.zeros([1, time_steps, 1])
    y = np.zeros([1, time_steps, 1])
    
    for group in df['group'].unique():
        temp_df = df[df['group'] == group]
        
        if len(temp_df) > time_steps:
            temp_x, temp_y = create_sequences(temp_df[[feature]], time_steps)
            x = np.concatenate((x, temp_x), axis=0)
            y = np.concatenate((y, temp_y), axis=0)
            valid_timestamp += temp_df['publishTimestamp'][time_steps - 1:].to_list()
            
    # Remove init padding zeros.
    x = x[1:]
    y = y[1:]

    return x, y, valid_timestamp