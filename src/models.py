import pandas as pd
import numpy as np
import warnings


class NormalModel:
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        
    def fit(self, Y, weights=None):
        '''fits the model parameters to an observation sequence, weights are optional'''
        # prepare
        Y = np.array(Y).reshape(-1, 1)
        if weights is None:
            weights = np.ones(Y.shape)
        else:
            weights = np.array(weights).reshape(-1, 1)
        
        # estimate mean
        mean = (Y*weights).sum(axis=0)/weights.sum(axis=0)
        
        # estimate variance
        errors = (Y-mean)**2
        variance = (errors*weights).sum(axis=0)/weights.sum(axis=0)
        
        # update
        self.mu = mean
        self.sigma = np.sqrt(variance)
        
    def pdf(self, Y):
        '''returns the likelihood of each observation in an observation sequence'''
        pdf = 1/(self.sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*(Y-self.mu)**2/self.sigma**2)
        return pdf
    
    def score(self, Y):
        '''returns the log-likelihood of an observation sequence'''
        score = np.log(self.pdf(Y)).sum()
        return score