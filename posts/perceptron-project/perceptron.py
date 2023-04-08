import numpy as np
from matplotlib import pyplot as plt

class Perceptron():

    def __init__(self):
        np.random.seed(123456)
        self.w = np.random.rand(3)
        self.history = []
        self.weight_history = [self.w]


    '''
    This function will take in an X array of features and
    a y array of labels. The purpose of this function is
    to adjust the weights of the perceptron model in order
    to accurately classify new points. In this function, to
    updates the weights of the perceptron, I implemented
    the equation in basic terms of:

     newWeights = oldWeights + (perceptronPrediction * realClassification) * currentDataPoint

    By using this equation, I am able to adjust the weights
    in the correct direction based on the performance of the
    perceptron model on this specific data point.

    '''
    def fit(self, X, y, max_steps):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        y_ = (2*y) - 1
        counter = 0
        score = 0
        while counter < max_steps and score != 1.0:
            ind = np.random.randint(0,len(X))
            point = X_[ind]
            y_hat = ((self.predict(point) >= 0) * 2) - 1
            y_hatY = (y_[ind] * y_hat < 0) * 1
            
            self.w = self.w + (y_hatY)*y_[ind]*X_[ind]
            self.weight_history.append(self.w)

            score = self.score(X, y)
            self.history.append(score)

            counter = counter + 1



    '''
    This function will take in an X array of features. 
    The purpose of this function is to make a prediction
    with the perceptron.
    '''
    def predict(self, X):
        return np.dot(X, self.w)

    
    '''
    Returns the current score of the perceptron object
    '''
    def score(self, X, y):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        y_ = (2*y) - 1
        return sum(abs(((((np.matmul(X_,self.w) > 0) * 2) - 1) + y_) / 2)) / len(y)

        
            