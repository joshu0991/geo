import numpy as np
from math import cos, sin
from pandas import DataFrame
from pandas import concat
from csv import reader

def load_file(file_name):
    data = list()
    labels = list()
    with open(file_name, 'r') as file:
        csv_reader = reader(file)
        for r in csv_reader:
            if not r:
                continue
            label = r[0]
            del r[0]
            data.append(r)
            labels.append(label)

    X = np.array(data)
    Y = np.array(labels)
    return X, Y

def load_file_clus(file_name):
    data = list()
    labels = list()
    with open(file_name, 'r') as file:
        csv_reader = reader(file)
        for r in csv_reader:
            if not r:
                continue
            id = r[0]
            long = r[1]
            lat = r[2]
            time = r[5]
            arr = [long, lat, time]
            data.append(arr)
            labels.append(id)

    X = np.array(data)
    Y = np.array(labels)
    return X, Y

def load_file_outliers(file_name):
    data = list()
    with open(file_name, 'r') as file:
        csv_reader = reader(file)
        for r in csv_reader:
            if not r:
                continue
            id = r[0]
            long = r[1]
            lat = r[2]
            time = r[3]
            arr = [long, lat, time]
            data.append(arr)
    X = np.array(data)
    return X

def load_file_outliers_space_time(file_name):
    data_X = list()
    data_Y = list()
    with open(file_name, 'r') as file:
        csv_reader = reader(file)
        for r in csv_reader:
            if not r:
                continue
            id = r[0]
            long = r[1]
            lat = r[2]
            time = r[3]
            arr = [long, lat]
            data_X.append(arr)
            data_Y.append(time)
    X = np.array(data_X)
    Y = np.array(data_Y)
    return X, Y

def load_file_outliers_real(file_name):
    data_X = list()
    data_Y = list()
    data_Z = list()
    with open(file_name, 'r') as file:
        csv_reader = reader(file)
        for r in csv_reader:
            if not r:
                continue
            time = r[0]
            long = r[1]
            lat = r[2]
            db_id = r[3]
            arr = [long, lat]
            data_X.append(arr)
            data_Y.append(time)
            data_Z.append(db_id)
    X = np.array(data_X)
    Y = np.array(data_Y)
    Z = np.array(data_Z)
    return (X, Y, Z)

def load_file_forcaster(file_name):
    data_X = []
    data_Y = list()
    data_Z = list()
    with open(file_name, 'r') as file:
        csv_reader = reader(file)
        for r in csv_reader:
            if not r:
                continue
            time = r[0]
            long = r[1]
            lat = r[2]

            tupe = convert_long_lat_to_space(long, lat)

            db_id = r[3]
            arr = [tupe[0], tupe[1], tupe[2]]
            data_X.append(arr)
            data_Y.append(float(time))
            data_Z.append(db_id)
    X = np.array(data_X)
    Y = np.array(data_Y)

    # No z since no label need atm
    # Z = np.array(data_Z)

    return (X, Y)

def convert_long_lat_to_space(long, lat):
    long = float(long)
    lat = float(lat)
    x = cos(lat) * cos(long)
    y = sin(lat) * sin(long)
    z = sin(lat)
    return (x, y, z)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

