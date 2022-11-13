import pandas as pd
import json
from flask import abort


def get_error_threshold(meta_dict, device_name, feature):
    '''
    Load the error_threshold from meta of the model to measure anomaly of a timestamp.
    '''
    try:
        error_threshold = meta_dict['model'][f'{device_name}_{feature}']['error_threshold']
    except KeyError:
        abort(422, description=f"Not available meta data for {device_name}: {feature} model.")

    return error_threshold


def get_min_max_error_index(loss_df):
    '''
    Find min and max error index to plot sample arrays.
    '''
    min_error_index = loss_df[loss_df['error'] == min(loss_df['error'])].index.to_list()[0]
    max_error_index = loss_df[loss_df['error'] == max(loss_df['error'])].index.to_list()[0]

    return min_error_index, max_error_index


def add_cross_ratio_flag(df, error_ratio_threshold):
    '''
    Indentify the timestamp that rolling_error cross threshold to record detected timestamp.
    '''
    if (df['error_ratio'] >= error_ratio_threshold) & (df['error_ratio_shift'] < error_ratio_threshold):
        cross_flag = 1
    else:
        cross_flag = 0
        
    return cross_flag


def get_error_ratio(df, rolling_interval, error_threshold, error_ratio_threshold):
    '''
    Calculate error_ratio from rolling percentage of anomaly timestamp and add cross flag.
    '''
    df['error_flag'] = 0
    df.loc[df['error'] > error_threshold, 'error_flag'] = 1
    df['error_count'] = df['error_flag'].rolling(rolling_interval).sum()

    try:
        df['error_ratio'] = df['error_count'] / rolling_interval
    except ZeroDivisionError:
        abort(422, description=f"rolling_error_days in configuration file must not be 0.")

    # df = df.dropna()
    df['error_ratio_shift'] = df['error_ratio'].shift()
    df['cross_flag'] = df.apply(add_cross_ratio_flag, error_ratio_threshold=error_ratio_threshold, axis=1)

    df = df.drop(columns=['error_flag', 'error_count', 'error_ratio_shift'])

    return df


def get_detection_df(loss_df, input_df, minute_interval, rolling_error_days, error_threshold, error_ratio_threshold):
    '''
    Get detection result from error_ratio.
    '''
    detection_df = pd.DataFrame({
        'publishTimestamp': pd.date_range(input_df.index[0], input_df.index[-1], freq=f'{minute_interval}min')
    })

    detection_df = detection_df.merge(loss_df, how='left', left_on='publishTimestamp', right_on='publishTimestamp')
    detection_df = detection_df.fillna(0)

    try:
        rolling_interval = int(((60 / minute_interval) * 24) * rolling_error_days)
    except ZeroDivisionError:
        abort(422, description=f"minute_interval in configuration file must not be 0.")

    detection_df = get_error_ratio(detection_df, rolling_interval, error_threshold, error_ratio_threshold)

    return detection_df
    

def get_error_flag(detection_df, error_ratio_threshold):
    '''
    Return status of a device from latest error_ratio.
    '''
    last_error_ratio = detection_df.loc[len(detection_df) - 1, 'error_ratio']
    
    if last_error_ratio >= error_ratio_threshold:
        error_flag = True
    else:
        error_flag = False

    return error_flag


def gen_status_dict():
    '''
    Generate status_dict to record result of each feature of each device in route predict_status.
    '''
    status_dict = {
        'deviceId': [],
        'feature': [],
        'status_message': []
    }

    return status_dict


def get_cross_timestamp(detection_df):
    '''
    Get all detected timestamp of a feature of a device.
    '''
    cross_df = detection_df[detection_df['cross_flag'] == 1].reset_index(drop=True)
    cross_list = []
    
    for i in range(len(cross_df)):
        cross_list.append(cross_df.loc[i, 'publishTimestamp'])

    return cross_list


def update_detected_timestamp(detection_df, model_path, device_name, feature, error_ratio_threshold):
    '''
    Update detected timestamp in model meta json.
    '''
    with open(f'{model_path}/meta.json') as meta_file:
        meta_dict = json.load(meta_file)

    error_flag = get_error_flag(detection_df, error_ratio_threshold)
    
    if not error_flag:
        meta_dict['model'][f'{device_name}_{feature}']['detected_timestamp'] = None
    else:
        cross_list = get_cross_timestamp(detection_df)

        if len(cross_list) > 0:
            # Record lastest cross as detected timestamp.
            # If no cross, error occur during past detection. Use the timestamp on model meta json.
            meta_dict['model'][f'{device_name}_{feature}']['detected_timestamp'] = str(cross_list[-1])
            
    with open(f'{model_path}/meta.json', 'w') as meta_file:
        json.dump(meta_dict, meta_file, indent=4)

    return meta_dict


def get_status_message(meta_dict, device_name, feature):
    detected_timestamp = meta_dict['model'][f'{device_name}_{feature}']['detected_timestamp']
    
    if detected_timestamp == None:
        status_message = "Normal"
    else:
        status_message = f"Detected timestamp: {detected_timestamp}"

    return status_message


def add_status_result(status_dict, meta_dict, device_name, device_id, feature):
    '''
    Add latest status of each feature of each device to status_dict in route predict_status.
    '''
    status_message = get_status_message(meta_dict, device_name, feature)

    status_dict['deviceId'].append(str(device_id))
    status_dict['feature'].append(feature)
    status_dict['status_message'].append(status_message)

    return status_dict


def clean_output(detection_df, input_df, feature):
    '''
    Clean the final output dataFrame to put in output_dict.
    '''
    def transform_iso_date(df):
        '''
        Transform date into ISO format.
        '''
        df['publishTimestamp'] = df['publishTimestamp'].apply(lambda x: str(x.strftime('%Y-%m-%dT%H:%M:%S.%f')))
        df['publishTimestamp'] = df['publishTimestamp'].apply(lambda x: x[:-3] + 'Z')

        return df

    output_df = detection_df.merge(input_df.reset_index(), how='left', left_on='publishTimestamp', right_on='publishTimestamp')
    output_df = output_df.rename(columns={feature:'raw_data'})
    output_df = output_df[['publishTimestamp', 'raw_data', 'error_ratio']]
    output_df = output_df.fillna(0)
    output_df = transform_iso_date(output_df)

    return output_df