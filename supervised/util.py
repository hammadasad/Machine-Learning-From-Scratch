import numpy as np
import pandas as pd

def get_data(limit = None):
    print "Reading MNIST data"
    dataFrame = pd.read_csv('./train.csv')
    data = dataFrame.as_matrix()
    np.random.shuffle(data)
    #inputs
    X = data[:, 1 :]
    #labels/classes
    Y = data[:, 0]
    # Setting a limit for amount of data returned
    if limit is not None:
        X, Y = X[: limit], Y[: limit]
    return X, Y
