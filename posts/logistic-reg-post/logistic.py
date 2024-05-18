import torch
import numpy as np

class LinearModel:

    def __init__(self):

        # here I intialize both my weight and the previous weight
        # which will be used in gradient descent
        self.previous = None
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
    # Here I define a sigmoid function for clarity
    def sigmoid(self, s):
        return 1 / (1  + torch.exp(-s))

    def loss(self, X, y):
        sigscore = self.sigmoid(self.score(X))
        
        # we calculate our loss
        l = torch.mean(-y * torch.log(sigscore) - (1 - y) * torch.log(1 - sigscore))
        return l

    def grad(self, X, y):
        # make vector v and then convert it to shape (n, 1) 
        v = self.sigmoid(self.score(X)) - y 
        v_ = v[:, None]

        # caculate the mean
        g = torch.mean(v_*X, dim = 0)
        return g 

class GradientDescentOptimizer(LinearModel):
    def __init__(self, model):
        self.model = model
    
    def step(self, X, y, alpha, beta):
                
        # if there is no previous weight, initialize a random weight
        if self.model.previous == None: 
            self.model.previous = torch.rand((X.size()[1]))

        # if there is no current weight, initialize a random weight
        if self.model.w == None:
            self.model.w = torch.rand((X.size()[1]))

        # we set a temp_weight to keep track of our current weight
        temp_weight = self.model.w

        # we update the current weight to a new weight
        self.model.w = self.model.w - alpha * self.model.grad(X, y) + beta * (self.model.w - self.model.previous)

        # we store our old current weight as a previous weight
        self.model.previous = temp_weight
