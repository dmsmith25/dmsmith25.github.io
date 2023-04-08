import numpy as np
from matplotlib import pyplot as plt

class LogisticRegression():

    def __init__(self):
        np.random.seed(123456)
        self.w = np.random.rand(3)
        self.loss_history = []
        self.score_history = []
        self.prev_loss = np.inf


    def sigmoid(self, z):
        return 1/(1+(np.exp(-z)))
    

    def logistic_loss(self, y_hat, y): 
        #print(np.log(self.sigmoid(y_hat)))
        return -y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))


    def empirical_risk(self, X, y, loss, w):
        y_hat = self.predict(X)
        return loss(y_hat, y).mean()

    
    def gradient(self, x, y):
        mult = (((self.sigmoid(self.predict(x)) - y) * x.T).T)
        return np.mean(mult, axis=0)
        

    '''
    This function will take in an X array of features and
    a y array of labels. The purpose of this function is to 
    use gradient descent to determine the weights. By the end
    of the functions runtime, the weights will reflect satisfactory
    for classifying new values of X. Simultaniously, this fit
    function will also calculate the loss of each iteration and 
    store these losses in loss_history. In this way, when we access
    loss_history, we can see the evolution of the models performance
    relative to each iteration.
    '''
    def fit(self, X, y, alpha, max_epochs):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)

        self.prev_loss = self.empirical_risk(X_, y, self.logistic_loss, self.w)
        counter = 0
        while counter < max_epochs and self.prev_loss != 0.0:

            self.w -= alpha*self.gradient(X_, y)  # gradient step
            new_loss = self.empirical_risk(X_, y, self.logistic_loss, self.w)

            self.loss_history.append(new_loss)

            # check if loss hasn't changed and terminate if so
            if np.isclose(new_loss, self.prev_loss):          
                counter = max_epochs
            else:
                self.prev_loss = new_loss

            counter = counter + 1

    
    '''
    This function is very similar to the fit function with a few modifications.
    In this fit_stochastic function, I implement stochastic gradient descent
    where instead of calculating the full gradient for the whole dataset, now we just need
    to take the gradient of a fraction of the dataset. The batch is chosen at random and
    the batch size is input from the user. Similar to the fit function, fit_stochastic
    computes the loss of the whole dataset for each iteration and stores it in loss_history.
    '''
    def fit_stochastic(self, X, y, alpha, m_epochs, batch_size):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        n = X.shape[0]
        arrangeList = np.arange(m_epochs)
        for j in arrangeList:
                    
            # randonmly deetermine batches
            order = np.arange(n)
            np.random.shuffle(order)

            for batch in np.array_split(order, n // batch_size + 1):


                x_batch = X[batch,:]
                y_batch = y[batch]

                X_batch = np.append(x_batch, np.ones((x_batch.shape[0], 1)), 1) # add 1 to all the vectors in batch

                grad = self.gradient(X_batch, y_batch) 
                self.w -= alpha*grad

            new_loss = self.empirical_risk(X_, y, self.logistic_loss, self.w)
            self.loss_history.append(new_loss)
            if np.isclose(new_loss, self.prev_loss):          
                break
            else:
                self.prev_loss = new_loss



    '''
    This function will take in an X array of features. 
    The purpose of this function is to make a prediction
    with the perceptron.
    '''
    def predict(self, X):
        return np.dot(X, self.w)

    
    '''
    Returns the current score of the perceptron object
    (self.predict(point) >= 0)
    '''
    def score(self, X, y):
        return sum(np.matmul(self.predict(X), y) > 0) / len(y)

    def loss(self, X, y):
        return 1 - self.score(X,y)