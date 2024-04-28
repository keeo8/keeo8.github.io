import torch
import numpy as np

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
        

        s = X@self.w
        return s
        
    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        s  = self.score(X)
        y_hat = (s > 0) * 1.0
        return y_hat

class LogisticRegression(LinearModel):
    def sigmoid(self, s):
        return 1 / (1  + torch.exp(-s))

    def loss(self, X, y):
        sigscore = self.sigmoid(self.score(X))
        l = torch.mean(-y * torch.log(sigscore) - (1 - y) * torch.log(1 - sigscore))
        return l

    def grad(self, X, y):
        # make vector v aand then convert it to shape (n, 1) 
        v = self.sigmoid(self.score(X)) - y 
        v_ = v[:, None]
        g = torch.mean(v_*X)
        return g 

class GradientDescentOptimizer(LinearModel):
    def __init__(self, model):
        self.model = model
        self.prev_weight = None
        self.current_weight = self.model.w
    
    def step(self, X, y, alpha, beta):
        
                
        if self.prev_weight == None: 
            self.prev_weight = torch.rand((X.size()[1]))

        if self.current_weight == None:
            self.current_weight = torch.rand((X.size()[1]))

        temp_weight = self.current_weight
        self.current_weight -= alpha * self.model.grad(X, y) + beta * (self.current_weight - self.prev_weight)
        self.model.w = self.current_weight
        self.prev_weight = temp_weight
    
