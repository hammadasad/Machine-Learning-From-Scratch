import numpy as np
import matplotlib.pyplot as plt

numDataPoints = 50

#50 Evenly Spaced Points from 0 to 10
X = np.linspace(0, 10, numDataPoints)

#Add random noise to Y
Y = 0.5 * X + np.random.randn(numDataPoints)

#Manually set outliers, set last and second last point to 30 bigger
Y[-1] += 30
Y[-2] += 30

plt.scatter(X, Y)
plt.show()

#Solve for Weights

#Add Bias Terms
X = np.vstack( [np.ones(numDataPoints), X]).T

#Max Likelihood Solution
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)
#Original Data
plt.scatter(X[:, 1], Y)
#Max Likelihood Line
plt.plot(X[:, 1], Yhat_ml)
plt.show()

#L2 Penalty
l2 = 1000

# weightMap = Inverse(Lambda*IdentityMatrix + Xtranspose*X) * Xtranspose*Y
w_map = np.linalg.solve(l2 * np.eye(2) + X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)
#Original Data
plt.scatter(X[:, 1], Y)
#Max Likelihood Line
plt.plot(X[:, 1], Yhat_ml, label = 'maximum Likelihood')
#MAP line
plt.plot(X[:, 1], Yhat_map, label = 'map')
plt.legend()
plt.show()
