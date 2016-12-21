#Predicting Systolic Blood Pressure

#data is from:
#http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3)/patient.
# X1 => systolic blood pressure
# X2 => age (years)
# X3 => weight (lbs)

#Requres Pandas to read the Excel Sheet => 'pip install xlrd'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataFrame = pd.read_excel('mlr02.xls')
X = dataFrame.as_matrix()

#Age vs Blood Pressure
plt.scatter(X[:, 1], X[:, 0])
plt.show()

#Weight vs Blood Pressure
plt.scatter(X[:, 2], X[:, 0])
plt.show()

#Partitioning
dataFrame['ones'] = 1
Y = dataFrame['X1']
X = dataFrame[['X2', 'X3', 'ones']]
x2_Only = dataFrame[['X2', 'ones']]
x3_Only = dataFrame[['X3', 'ones']]

#R^2 Calculate
def get_r2(X, Y):
    weight = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Yhat = X.dot(weight)

    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    return r2

print("R2 for Age Only is : ", get_r2(x2_Only, Y))
print("R2 for Weight Only is : ", get_r2(x3_Only, Y))
print("R2 for both Age & Weight is : ", get_r2(X, Y))
