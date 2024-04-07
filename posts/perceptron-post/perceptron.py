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

class Perceptron(LinearModel):
    def loss(self, X, y):
        """
        Compute the misclassification rate. In the perceptron algorithm, the target vector y is assumed to have labels in {-1, 1}. A point i is classified correctly if its score s_i has the same sign as y_i. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). In the perceptron algorithm, the possible labels for y are assumed to be {-1, 1}
            HINT: In order to use the math formulas in the lecture, you are going to need to construct a modified set of targets and predictions that have entries in {-1, 1} -- otherwise none of the formulas will work right! An easy to to make this conversion is: 
        
             y_ = 2*y - 1

        """
        s = self.score(X) 
        updated = (s * y) > 0
        return updated.float().mean()

    def grad(self, X, y):
        s = self.w * X
        grad_v = torch.where(torch.mul(s, y) < 0, y * X, torch.zeros_like(X))
        return grad_v

class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """
        self.model.loss(X, y)
        self.model.w = self.model.w + self.model.grad(X, y)
        pass