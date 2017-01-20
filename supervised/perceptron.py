import numpy as np
import matplotlib.pyplot as plt
from util import get_data as MNIST
from datetime import datetime

# Create linearly seperable data that we can plot since it'll be 2D
def get_data():
    weight = np.array([-0.5, 0.5])
    bias = 0.1
    # Make it uniformly distributed between -1 & +1
    X = np.random.random((300, 2)) * 2 - 1
    Y = np.sign(X.dot(weight) + bias)
    return X, Y

class Perceptron:
    def fit(self, X, Y, learning_rate = 1.0, epochs = 1000):
        dimensionality = X.shape[1]
        self.weight = np.random.randn(dimensionality)
        self.bias = 0

        numSamples = len(Y)
        costs = []

        for epoch in xrange(epochs):
            Yhat = self.predict(X)
            incorrect = np.nonzero(Y != Yhat)[0]
            if len(incorrect) == 0:
                break
            randomIncorrectSample = np.random.choice(incorrect)
            # Update
            self.weight += learning_rate * Y[randomIncorrectSample] * X[randomIncorrectSample]
            self.bias += learning_rate * Y[randomIncorrectSample]

            #Incorrect rate
            cost = len(incorrect) / float(numSamples)
            costs.append(cost)

        print "final weight: ", self.weight, "final bias: ", self.bias, "epochs: ", (epoch + 1), "/", epochs

        plt.plot(costs)
        plt.show()

    def predict(self, X):
        return np.sign(X.dot(self.weight + self.bias))

    def score(self, X, Y):
        probability = self.predict(X)
        return np.mean(probability == Y)


if __name__ == '__main__' :
    X, Y = get_data()
    plt.scatter(X[:, 0], X[:, 1], c = Y, s = 100, alpha = 0.5 )
    plt.show()

    numTrain = len(Y) / 2
    Xtrain, Ytrain = X[:numTrain], Y[:numTrain]
    Xtest, Ytest = X[numTrain:], Y[numTrain:]

    perceptronModel = Perceptron()

    #Train
    time0 = datetime.now()
    perceptronModel.fit(Xtrain, Ytrain)
    print "Time it took to train: ", (datetime.now() - time0)

    #Score
    time0 = datetime.now()
    print "Train accuracy is: ", perceptronModel.score(Xtrain, Ytrain)
    print "Time it took to get train accuracy is : ", (datetime.now() - time0), " with train size : ", len(Ytrain)

    #Test the test data
    time0 = datetime.now()
    print "Test Accuracy is:", perceptronModel.score(Xtest, Ytest)
    print "Time it took to get the test accuacy is: ", (datetime.now() - time0), " with test size: ", len(Ytest)

    # Goes to about 1000 epochs (iterations)
