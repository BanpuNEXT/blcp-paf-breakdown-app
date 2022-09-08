import numpy as np
import pandas as pd
import datetime as dt
from tensorflow import keras
import json
from flask import abort


def gen_model(x):
    '''
    Construct AutoEncoder layers.
    '''
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(x.shape[1], x.shape[2])),
            keras.layers.Conv1D(
                filters=32, kernel_size=7, padding='same', strides=2, activation='relu'
            ),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Conv1D(
                filters=16, kernel_size=7, padding='same', strides=2, activation='relu'
            ),
            keras.layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding='same', strides=2, activation='relu'
            ),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding='same', strides=2, activation='relu'
            ),
            keras.layers.Conv1DTranspose(filters=x.shape[2], kernel_size=7, padding='same'),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    
    return model


def train_model(x, y, model, model_path, device_name, feature):
    '''
    Train and save the model of a feature of a device.
    '''
    history = model.fit(
        x,
        y,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
        ],
    )

    model.save(f'{model_path}/{device_name}_{feature}.h5')
    
    return history


def create_model_meta(error_threshold, model_path, device_name, feature):
    '''
    Save error_threshold and training_timestamp as meta for the model.
    '''
    with open(f'{model_path}/meta.json') as meta_file:
        meta_dict = json.load(meta_file)

    
    timestamp_str = str(dt.datetime.now())

    if f'{device_name}_{feature}' not in meta_dict['model']:
        meta_dict['model'][f'{device_name}_{feature}'] = {
            'error_threshold': 0,
            'training_timestamp':0
        }

    meta_dict['model'][f'{device_name}_{feature}']['error_threshold'] = error_threshold
    meta_dict['model'][f'{device_name}_{feature}']['training_timestamp'] = timestamp_str

    with open(f'{model_path}/meta.json', 'w') as meta_file:
        json.dump(meta_dict, meta_file, indent=4)


def cal_loss(x_pred, x, error_direction):
    '''
    Calculate loss from error between input and reconstruction form the model.
    '''
    if error_direction == 'upper':
        # The more x, the more error.
        batch_loss = np.where(x < x_pred, 0, x - x_pred)
        loss = np.mean(batch_loss, axis=1)
    elif error_direction == 'lower':
        # The less x, the more error.
        batch_loss = np.where(x_pred < x, 0, x_pred - x)
        loss = np.mean(batch_loss, axis=1)
    else:
        # Take both side into account.
        loss = np.mean(np.abs(x_pred - x), axis=1)

    if x.shape[2] > 1:
        # If multi variables model, calculate mean between features.
        loss = np.mean(loss, axis=1)

    return loss


def cal_error_threshold(training_loss_df, min_error_threshold=0.05):
    '''
    Set max training error to be threshold of a model with minimum value 0.05. 
    '''
    error_threshold = max(np.max(training_loss_df['error']), min_error_threshold)

    return error_threshold


def get_reconstruct_loss(loss, valid_timestamp):
    '''
    Assign timestamp to reconstruction loss from valid_timestamp by function group_sequence.
    '''
    loss_df = pd.DataFrame({
        'publishTimestamp': valid_timestamp,
        'error': loss.reshape(1, -1)[0]
    })

    return loss_df


def load_model(model_path, device_name, feature):
    '''
    Load the model of a feature of a device.
    '''
    try:
        model = keras.models.load_model(f'{model_path}/{device_name}_{feature}.h5')
    except OSError:
        abort(422, description=f"Not available model for {device_name}: {feature}.")

    return model