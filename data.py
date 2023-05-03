from sklearn import preprocessing
import pandas as pd
import numpy as np
import torch

def read_timeseries(path, skiprows, sampling_period, time_column, label_column):
    df = pd.read_csv(path, index_col=None, low_memory=False)
    # sampling
    df = df.iloc[skiprows::sampling_period, :]
    # timestampes, labels
    ts, labels = df.iloc[:, time_column].tolist(), df.iloc[:, -1].to_numpy()
    # select numeric columns only
    df = df.iloc[:, time_column+1:label_column]
    data = df.to_numpy(dtype=np.float64)
    return data, ts, labels

def read_data(path, skiprows, sampling_period, \
    time_column, label_column, normalizer, n_window):
    data, ts, labels = read_timeseries(path, skiprows, \
        sampling_period, time_column, label_column)
    # normalize
    if normalizer is None:
        normalizer = preprocessing.StandardScaler()
        data = normalizer.fit_transform(data)
    else:
        data = normalizer.transform(data)
    # convert to windows
    data = np.lib.stride_tricks.sliding_window_view(data, n_window, axis=0)
    # copy data to avoid memory issues
    data = np.copy(data)
    data = np.transpose(data, (0, 2, 1))
    # convert to torch tensor
    data = torch.from_numpy(data).double()
    ts, labels = ts[n_window-1:], labels[n_window-1:]
    assert len(data) == len(ts) == len(labels)
    return data, ts, labels, normalizer

