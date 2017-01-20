# Implementation of KNN (K Nearest Neighbours)
# K is a hyperparameter, we'll use 1...5, but K is generally Z^+

# Premise: To classify, we use the closest known data points
# Keeping track of an abitrary number of distances

# 'sudo pip install sortedcontainers' to use SortedList
# Can't use Sorted Dictionary as we don't want same distances to be overwritten

import numpy as np
from sortedcontainers import SortedList

from util import get_data
from datetime import datetime

class KNN(object):
    def __init__(self, k):
        self.k = k;

    #training method - it's a lazy classifier
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        for index, x in enumerate(X):
            sortedList = SortedList(load = self.k)
            for index2, trainingPoint in enumerate(self.X):
                difference = x - trainingPoint
                # Doesn't matter if we use Euclidean or Squared Distance. Square func is monotically increasing
                distance = difference.dot(difference)
                if len(sortedList) < self.k:
                    sortedList.add((distance, self.y[index2]))
                else:
                    if distance < sortedList[-1][0]:
                        del sortedList[-1]
                        sortedList.add((distance, self.y[index2]))
            votes = {}
            for _ , aVote in sortedList:
                votes[aVote] = votes.get(aVote, 0) + 1
            max_votes = 0
            max_votes_class = -1
            for aVote, count in votes.iteritems():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = aVote
            y[index] = max_votes_class
        return y
    def score(self, X, Y):
        prediction = self.predict(X)
        return np.mean(prediction == Y)

if __name__ == '__main__':
    X, Y = get_data(2000)
    numPointsTrain = 1000
    Xtrain, Ytrain = X[:numPointsTrain], Y[:numPointsTrain]
    Xtest, Ytest = X[numPointsTrain :], Y[numPointsTrain:]
    for k in (1, 2, 3, 4, 5, 6):
        knn = KNN(k)
        timeStart = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print("The training time is: ", datetime.now() - timeStart)

        timeStart = datetime.now()
        print("Before predicting, the time is: ", timeStart)
        print("The training accuracy is : ", knn.score(Xtrain, Ytrain))
        print("Time it took training is: ", datetime.now() - timeStart)

        timeStart = datetime.now()
        print("Before predicting, the time is: ", timeStart)
        print("The training accuracy is : ", knn.score(Xtest, Ytest))
        print("Time it took training is: ", datetime.now() - timeStart)
