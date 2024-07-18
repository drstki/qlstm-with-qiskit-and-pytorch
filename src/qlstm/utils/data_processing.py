import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch


def data_processing(input_data: pd.DataFrame, time_sequence: int = 4, train_test_split: float = 0.75, reshape_inputs: bool = True):
    """
    Creates a time_series matrix and target array and splits these into training and test sets for sequence prediction tasks.

    Parameters:
        input_data (pd.DataFrame, mandatory): The input data to be processed and split.
        time_sequence (int, optional): The N time steps in each sequence (default is 4).
        split_size (float, optional): The proportion of data to be allocated for training (default is 0.75).
        reshape_inputs (bool, optional): Parameter decides if the x_data (inputs) should be reshaped to (batch_size, time_window, features)

    Returns:
        x_train (torch.Tensor): The training input data.
        y_train (torch.Tensor): The training target data.
        x_test (torch.Tensor): The testing input data.
        y_test (torch.Tensor): The testing target data.

    Raises:
        Exception: If the shapes of the data arrays are not consistent.
    """

    ### Preprocess the data ###
    # convert to numpy array
    data = input_data.values
    # save unprocessed y_data
    y_data = data
    # reshape the data from (x, 1) to (x,)
    data = np.squeeze(data)
    # create a sliding window view
    data = sliding_window_view(data.T, time_sequence)
    # get the sequence data
    x_data = data[:-1]
    # get the target data (the next value)
    y_data = y_data[time_sequence:]

    ### Split the data into train and test set  ###
    # cast to int (round implicitly down)
    train_number = int(len(x_data) * train_test_split)
    train_number
    # split the data
    x_train = x_data[:train_number]
    y_train = y_data[:train_number]
    x_test = x_data[train_number:]
    y_test = y_data[train_number:]

    ### Validate the shapes ###
    if x_data.shape[0] == y_data.shape[0] and x_train.shape[0] == y_train.shape[0] and x_test.shape[0] == y_test.shape[0]:
        print("All shapes are correct.")
    else:
        # compose error message
        message = ("\nx_data.shape: " + str(x_data.shape) + "\ny_data.shape: " + str(y_data.shape)
                    + "\nx_train.shape: " + str(x_train.shape) + "\ny_train.shape: " + str(y_train.shape)
                    + "\nx_test.shape: " + str(x_test.shape) + "\ny_test.shape: " + str(y_test.shape)
                    + "\n\nx_data and y_data - x_train and y_train - x_test and y_test,\nshould have the same number of rows!")
        raise Exception(message)

    if reshape_inputs:
        # reshape the from (batch_size, time_window) to (batch_size, time_window, features)
        # features = 1 for univariate time series
        x_train = torch.from_numpy(x_train.copy())
        x_train = torch.unsqueeze(x_train, dim=2) 

        x_test = torch.from_numpy(x_test.copy())
        x_test = torch.unsqueeze(x_test, dim=2)  

        y_train = torch.from_numpy(y_train.copy())
        y_train = torch.unsqueeze(y_train, dim=2)

        y_test = torch.from_numpy(y_test.copy())
        y_test = torch.unsqueeze(y_test, dim=2)
    else:
        x_train = torch.from_numpy(x_train.copy())
        x_test = torch.from_numpy(x_test.copy())
        y_train = torch.from_numpy(y_train.copy())
        y_test = torch.from_numpy(y_test.copy())

    # cast to float32, otherwise the model will not work with float64
    return x_train.to(torch.float32), y_train.to(torch.float32), x_test.to(torch.float32), y_test.to(torch.float32)


def shift_train_test_predict(data: pd.DataFrame, train_predict, test_predict,  time_sequence: int = 4):
    """
    This function shifts the predicted values for the training and test set, so that they can be plotted together with the original data.
    (for visualization purposes only)

    Parameters:
        train_predict (np.ndarray): Predicted values for the training set.
        test_predict (np.ndarray): Predicted values for the test set.
        data (pd.DataFrame): Original data.
        time_sequence (int): The N time steps in each sequence (default is 4).

    Returns:
        train_predict_plot: A copy of the original data with NaN values, where the predicted values for the training set are inserted.
        test_predict_plot: A copy of the original data with NaN values, where the predicted values for the test set are inserted.
    """

    # shift train predictions for plotting
    train_predict_plot = data.copy().values
    train_predict_plot[:] = np.NaN
    train_predict_plot[time_sequence : len(train_predict) + time_sequence, :] = train_predict
    # shift test predictions for plotting
    test_predict_plot = data.copy().values
    test_predict_plot[:] = np.NaN
    test_predict_plot[time_sequence + len(train_predict) -1 : - 1, :] = test_predict

    return train_predict_plot, test_predict_plot

def time_window(input_data, window):
    window_data = []
    L = len(input_data)
    for i in range(L-window):
        train_series = input_data[i:i+window]
        train_label = input_data[i+window:i+window+1,0]
        # train_label = input_data[i+1:i+window+1,0]
        window_data.append((train_series ,train_label))
    return window_data

def time_window_batch(input_data, window, batch_size):
    window_data = []
    L = len(input_data)
    for i in range(0, L - window, batch_size):
        train_series_batch = []
        train_label_batch = []
        for j in range(batch_size):
            train_series = input_data[i+j:i+j+window]
            train_label = input_data[i+j+window:i+j+window+1, 0]
            train_series_batch.append(train_series)
            train_label_batch.append(train_label)
        window_data.append((np.array(train_series_batch), np.array(train_label_batch)))
    return window_data
