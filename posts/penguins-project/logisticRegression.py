import numpy as np
from matplotlib import pyplot as plt

class LogisticRegression():

    def __init__(self):
        np.random.seed(123456)
        self.w = np.random.rand(2)
        self.loss_history = []
        self.score_history = []
        self.prev_loss = np.inf

    
    def sigmoid(self, z):
        return 1/(1-(np.exp(-z)))

    def logistic_loss(self, y_hat, y): 
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))

    def empirical_risk(self, X, y, loss, w):
        y_hat = self.predict(X, w)
        return loss(y_hat, y).mean()

    def gradient(self, x, y):
        return np.dot((self.sigmoid(self.predict(x, self.w)) - y),x)


    '''
    This function will take in an X array of features and
    a y array of labels. The purpose of this function is
    to find the seperation line
    '''
    def fit(self, X, y, alpha, max_epochs):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        y_ = (2*y) - 1
        counter = 0
        score = 0
        while counter < max_epochs and self.prev_loss != 0.0:
            ind = np.random.randint(0,len(X))
            point = X_[ind]
            y_hat = ((self.predict(point) >= 0) * 2) - 1
            y_hatY = (y_[ind] * y_hat < 0) * 1

            self.w -= alpha*self.gradient(self.w, X_, y)                      # gradient step
            new_loss = self.empirical_risk(X_, y, self.logistic_loss, self.w)
            
            self.w = self.w + (y_hatY)*y_[ind]*X_[ind]

            self.history.append(new_loss)
            # check if loss hasn't changed and terminate if so
            if np.isclose(new_loss, prev_loss):          
                counter = max_epochs
            else:
                prev_loss = new_loss

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
        return sum(abs(((((np.matmul(X,self.w) > 0) * 2) - 1) + y) / 2)) / len(y)

    #def loss(self, X, y):


        
            