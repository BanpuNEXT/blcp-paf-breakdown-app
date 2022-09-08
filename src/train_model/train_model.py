import pymongo
import json
import sys
import os

src_path = '..'
model_path = f'{src_path}/model'
scaler_path = f'{src_path}/scaler'
sys.path.append(os.path.abspath(src_path))

import func.func_data as func_data
import func.func_model as func_model


with open(f'{src_path}/secret/secret.json') as secret_file:
    secret_dict = json.load(secret_file)
    
with open(f'{src_path}/train_model/config.json') as config_file:
    config_dict = json.load(config_file)

client = pymongo.MongoClient(secret_dict['mongo_url'])
db = client['blcp-poc']


for device_name in config_dict['device_name']:
    for feature in config_dict['feature']:
        print(f'Device: {device_name}\nFeature: {feature}\n')
        
        device_id = func_data.get_device_id(device_name)
        start_date, end_date = func_data.get_training_date(
            start_training_date=config_dict['start_training_date'],
            n_training_day=config_dict['n_training_day'],
            n_test_day=config_dict['n_test_day'],
            )

        query = func_data.get_query(device_id, start_date, end_date)
        input_df = func_data.load_data(query, db)

        input_df = func_data.clean_type(input_df)
        input_df = func_data.add_additional_feature(input_df)
        input_df = func_data.filter_data(input_df, config_dict['low_generation_threshold'])
        input_df = func_data.filter_feature(input_df, feature)

        training_df, test_df, _ = func_data.split_data(
            input_df,
            config_dict['n_training_day'],
            config_dict['n_look_back_day']
            )

        training_df = func_data.divide_non_operation(training_df, config_dict['minute_interval'])
        test_df = func_data.divide_non_operation(test_df, config_dict['minute_interval'])

        training_df, test_df = func_data.transform_scale(
            training_df,
            test_df,
            scaler_path,
            device_name,
            feature
            )

        time_steps = func_data.cal_time_steps(config_dict['minute_interval'], config_dict['n_look_back_day'])
        x_train, y_train, training_valid_timestamp = func_data.group_sequence(training_df, feature, time_steps)
        x_test, y_test, test_valid_timestamp = func_data.group_sequence(test_df, feature, time_steps)

        model = func_model.gen_model(x_train)
        history = func_model.train_model(x_train, y_train, model, model_path, device_name, feature)

        x_train_pred = model.predict(x_train)
        training_loss = func_model.cal_loss(
            x_train_pred,
            x_train,
            error_direction=config_dict['feature'][feature]['error_direction']
        )
        
        training_loss_df = func_model.get_reconstruct_loss(training_loss, training_valid_timestamp)
        error_threshold = func_model.cal_error_threshold(training_loss_df)
        func_model.create_model_meta(error_threshold, model_path, device_name, feature)