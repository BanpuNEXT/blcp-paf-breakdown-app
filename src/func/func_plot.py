import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_data(training_df, test_df, feature, split_date):
    '''
    Plot timeseries of the feature of training and test set.
    '''
    plt_df = pd.concat([training_df, test_df])

    plt.figure(figsize=(20, 8))
    plt.plot(plt_df.index, plt_df[feature])
    plt.axvspan(training_df.index[0], split_date, color='grey', alpha=0.2)
    plt.xlim(min(training_df.index), max(training_df.index))
    plt.title('Raw data')
    plt.show()


def plot_predict_data(pred_df, feature):
    '''
    Plot timeseries of the feature of predict set.
    '''
    plt.figure(figsize=(20, 8))
    plt.plot(pred_df.index, pred_df[feature])
    plt.xlim(min(pred_df.index), max(pred_df.index))
    plt.title('Raw data')
    plt.show()


def plot_training_loss(history):
    '''
    Plot training and validating loss progress.
    '''
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()


def plot_predict_loss(loss):
    '''
    Plot min and max loss between input and reconstruction training set.
    '''
    min_error = np.min(loss)
    max_error = np.max(loss)

    print(f"Min validate error: {min_error}")
    print(f"Max validate error: {max_error}")

    plt.hist(loss)
    plt.xlabel("Loss")
    plt.ylabel("No of samples")
    plt.show()


def plot_min_max_error(x, x_pred, min_error_index, max_error_index):
    '''
    Plot reconstruction of min and max loss.
    '''
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    
    ax[0].plot(x[min_error_index])
    ax[0].plot(x_pred[min_error_index])
    ax[0].set_title("Min error")
    
    ax[1].plot(x[max_error_index])
    ax[1].plot(x_pred[max_error_index])
    ax[1].set_title("Max error")
    
    fig.supxlabel("Index")
    fig.supylabel("Value")


def plot_error_trend(detection_df, error_threshold, split_date=None):
    '''
    Plot series of reconstruction loss.
    '''
    plt.figure(figsize=(20, 8))

    plt.plot(detection_df['publishTimestamp'], detection_df['error'])
    plt.axhline(y=error_threshold, color='r', linestyle='-')

    if split_date != None:
        plt.axvspan(detection_df['publishTimestamp'][0], split_date, color='grey', alpha=0.2)
    
    plt.xlim(min(detection_df['publishTimestamp']), max(detection_df['publishTimestamp']))
    plt.ylim(bottom=0)
    plt.title("Error trend")


def plot_error_ratio(detection_df, error_ratio_threshold, split_date=None):
    '''
    Plot error_ratio from rolling percentage of anomaly timestamp to identify final result.
    '''
    plt.figure(figsize=(20, 8))

    plt.plot(detection_df['publishTimestamp'], detection_df['error_ratio'])
    plt.axhline(y=error_ratio_threshold, color='r', linestyle='-')

    if split_date != None:
        plt.axvspan(detection_df['publishTimestamp'][0], split_date, color='grey', alpha=0.2)

    plt.xlim(min(detection_df['publishTimestamp']), max(detection_df['publishTimestamp']))
    plt.ylim(0, 1)
    plt.title("Error ratio")