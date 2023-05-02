import numpy as np
class LinearRegression:

    def __init__(self, num_features= 1):
        np.random.seed(123456)
        self.feats = num_features + 1
        self.w = np.random.rand(num_features + 1)
        self.loss_history = []
        self.score_history = []
        self.prev_loss = np.inf

    def fit_analytic(self, X, y):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        self.w = np.linalg.inv(X_.T@X_)@X_.T@y

    def fit_gradient(self, X, y, alpha, max_iter):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        P = X_.T@X_
        q = X_.T@y

        for i in range(max_iter):
            gradient = 2*((P@self.w) - q)
            self.w -= alpha*gradient

            new_score = self.score(X,y)

            self.score_history.append(new_score)


    def predict(self, X):
        return X@self.w

    def score(self, X, y):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        y_bar = (1/len(y)) * np.sum(y)

        return 1 - ((np.sum(np.square(self.predict(X_) - y))) / (np.sum(np.square(y_bar - y))))
    
