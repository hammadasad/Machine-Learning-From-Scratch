import numpy as np
import pandas as pd

def get_data():
    dataFrame = pd.read_csv('ecommerce_data.csv')
    #dataFrame.head()

    X = data[:, :-1]
    Y = data[:, -1]

    # Normalize Numerical Columns
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

    # Categoral Column - Time of Day
    numberDataPoints, dimensionality = X.shape
    # 4 Different Categoral Values
    X2 = np.zeros((numberDataPoints, dimensionality + 3))
    X2[:, 0 : (dimensionality - 1)] = X[:, 0 : (dimensionality - 1)]

    # One-hot Encoding
    for n in xrange(numberDataPoints):
        timeOfDay = int(X[n, dimensionality - 1])
        X2[n, timeOfDay + dimensionality - 1] = 1

    return X2, Y
