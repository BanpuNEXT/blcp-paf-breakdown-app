from flask import Flask, request
import pymongo
from werkzeug.local import LocalProxy
import pandas as pd
from bson.objectid import ObjectId
import json
import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

src_path = '.'
model_path = f'{src_path}/model'
scaler_path = f'{src_path}/scaler'
sys.path.append(os.path.abspath(src_path))

import func.func_data as func_data
import func.func_model as func_model
import func.func_result as func_result


with open('secret/secret.json') as secret_file:
    secret_dict = json.load(secret_file)

app = Flask(__name__)
app.config['MONGO_URI'] = secret_dict['mongo_url']
app.config['JSON_SORT_KEYS'] = False


def get_config_meta_dict():
    '''
    Get model configuation and model training meta.
    '''
    with open(f'{src_path}/train_model/config.json') as config_file:
        config_dict = json.load(config_file)

    with open(f'{src_path}/model/meta.json') as meta_file:
        meta_dict = json.load(meta_file)

    return config_dict, meta_dict


def get_db():
    '''
    Get database object.
    '''
    def connect_db():
        '''
        Configuration method to return db instance.
        '''
        client = pymongo.MongoClient(secret_dict['mongo_url'])
        db = client['blcp-poc']
        
        return db
    
    print("GET Database")
    db = LocalProxy(connect_db)

    return db


def get_args():
    '''
    GET device_id and feature argement.
    '''
    args = request.args
    device_id = args.get('deviceId')
    feature = args.get('feature')

    return device_id, feature


def generate_status_output(status_dict):
    '''
    Reshape status dict output into array dict.
    '''
    output_data = {
        'status': pd.DataFrame(status_dict).to_dict('record'),
        }

    return output_data


def generate_error_ratio_dict(output_df, config_value_list):
    '''
    Reshape error_ratio dataFrame output into array dict.
    '''
    output_data = {
        'error_ratio': output_df.to_dict('records'),
        'config_value': config_value_list
        }

    return output_data


def get_process_time(start_time):
    '''
    Get process time in milliseconds.
    '''
    end_time = time.time()
    process_time = (end_time - start_time) * 1000

    return process_time


def gen_output_format(output_data, process_time):
    '''
    POST output in json format.
    '''
    output_dict = {
        'code': 0,
        'message': "success",
        'time': process_time,
        'data': output_data
    }

    return output_dict


@app.errorhandler(400)
def send_error(error):
    '''
    Send error for input error.
    '''
    error_dict = {
        'code': 400,
        'message': error.description
    }

    return error_dict


@app.errorhandler(422)
def send_error(error):
    '''
    Send error for app error.
    '''
    error_dict = {
        'code': 422,
        'message': error.description
    }

    return error_dict


@app.errorhandler(500)
def send_error(error):
    '''
    Send error for server error.
    '''
    error_dict = {
        'code': 500,
        'message': error.description
    }

    return error_dict


@app.route('/healthz')
def health():
    '''
    Get sample data to check database connection.
    Fix parameter as following
        - device_id: for PAF_1A
        - start_date, end_date: latest 90 days on database
    '''
    start_time = time.time()
    
    db = get_db()
    device_id = ObjectId('62ff07274adcc50b51965f96')
    start_date, end_date = func_data.get_predict_date(db)
    print(start_date, end_date)
    
    query = func_data.get_query(device_id, start_date, end_date)
    func_data.load_data(query, db)
    
    process_time = get_process_time(start_time)
    
    if process_time < 5000:
        health_dict = {
            'code': 200,
            'message': 'Everything is OK'
        }
    else:
        health_dict = {
            'code': 202,
            'message': 'Take too much time to connect to database.'
        
        }
    return health_dict
    

@app.route('/predict/paf/status')
def predict_status():
    '''
    API to get status of all feature of all device.
    '''
    start_time = time.time()
    
    config_dict, meta_dict = get_config_meta_dict()
    db = get_db()
    status_dict = func_result.gen_status_dict()
    start_date, end_date = func_data.get_predict_date(db)
    
    for device_name in config_dict['device_name']:
        for feature in config_dict['feature']:
            print(f"Device: {device_name}\nFeature: {feature}")

            print("   GET input")
            device_id = func_data.get_device_id(device_name)

            query = func_data.get_query(device_id, start_date, end_date)
            input_df = func_data.load_data(query, db)

            print("   Clean data")
            input_df = func_data.clean_type(input_df)
            input_df = func_data.add_additional_feature(input_df)
            input_df = func_data.filter_data(input_df, config_dict['low_generation_threshold'])
            input_df = func_data.filter_feature(input_df, feature)
            group_df = func_data.divide_non_operation(input_df, config_dict['minute_interval'])
            group_df = func_data.load_scale(group_df, scaler_path, device_name, feature)

            time_steps = func_data.cal_time_steps(config_dict['minute_interval'], config_dict['n_look_back_day'])
            x, _, valid_timestamp = func_data.group_sequence(group_df, feature, time_steps)

            print("   Load model")
            model = func_model.load_model(model_path, device_name, feature)

            print("   Predict")
            x_pred = model.predict(x)

            print("   Cal loss")
            predict_loss = func_model.cal_loss(
                x_pred,
                x,
                error_direction=config_dict['feature'][feature]['error_direction']
            )

            predict_loss_df = func_model.get_reconstruct_loss(predict_loss, valid_timestamp)
            error_threshold = func_result.get_error_threshold(meta_dict, device_name, feature)

            print("   Detect breakdown")
            detection_df = func_result.get_detection_df(
                predict_loss_df,
                input_df,
                minute_interval=config_dict['minute_interval'],
                rolling_error_days=config_dict['rolling_error_days'],
                error_threshold=error_threshold,
                error_ratio_threshold=config_dict['error_ratio_threshold']
            )

            status_dict = func_result.add_status_result(
                status_dict,
                detection_df,
                device_id,
                feature,
                error_ratio_threshold=config_dict['error_ratio_threshold']
                )

    output_data = generate_status_output(status_dict)    
    process_time = get_process_time(start_time)
    output_dict = gen_output_format(output_data, process_time)

    print("   Done\n")

    return output_dict


@app.route('/predict/paf/error_ratio')
def predict_error_ratio():
    '''
    API to get error_ratio of given feature of given device.
    '''
    start_time = time.time()
    
    config_dict, meta_dict = get_config_meta_dict()
    db = get_db()
    device_id, feature = get_args()
    device_name = func_data.get_device_name(device_id)
    start_date, end_date = func_data.get_predict_date(db)
    
    print("   GET input")
    # Convert str device_id from arg to ObjectId.
    device_id = func_data.get_device_id(device_name)
    query = func_data.get_query(device_id, start_date, end_date)
    input_df = func_data.load_data(query, db)

    print("   Clean data")
    input_df = func_data.clean_type(input_df)
    input_df = func_data.add_additional_feature(input_df)
    input_df = func_data.filter_data(input_df, config_dict['low_generation_threshold'])
    input_df = func_data.filter_feature(input_df, feature)
    group_df = func_data.divide_non_operation(input_df, config_dict['minute_interval'])
    group_df = func_data.load_scale(group_df, scaler_path, device_name, feature)

    time_steps = func_data.cal_time_steps(config_dict['minute_interval'], config_dict['n_look_back_day'])
    x, _, valid_timestamp = func_data.group_sequence(group_df, feature, time_steps)

    print("   Load model")
    model = func_model.load_model(model_path, device_name, feature)

    print("   Predict")
    x_pred = model.predict(x)

    print("   Cal loss")
    predict_loss = func_model.cal_loss(
        x_pred,
        x,
        error_direction=config_dict['feature'][feature]['error_direction']
    )

    predict_loss_df = func_model.get_reconstruct_loss(predict_loss, valid_timestamp)
    error_threshold = func_result.get_error_threshold(meta_dict, device_name, feature)

    print("   Detect breakdown")
    detection_df = func_result.get_detection_df(
        predict_loss_df,
        input_df,
        minute_interval=config_dict['minute_interval'],
        rolling_error_days=config_dict['rolling_error_days'],
        error_threshold=error_threshold,
        error_ratio_threshold=config_dict['error_ratio_threshold']
    )

    print("   Generate final output\n")            
    output_df = func_result.clean_output(detection_df, input_df, feature)

    config_value_list = [{
        'error_ratio_threshold':config_dict['error_ratio_threshold']
        }]
    output_data = generate_error_ratio_dict(output_df, config_value_list)
    process_time = get_process_time(start_time)
    output_dict = gen_output_format(output_data, process_time)

    print("   Done\n")

    return output_dict