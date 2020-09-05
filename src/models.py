import numpy as np
import warnings
from scipy.stats import norm


class BaseModel:
    
    '''
    Base class for models
    '''
    
    def __init__(self):
        pass
    
    def score(self, Y):
        
        '''
        Returns the log-likelihood of an observation sequence
        '''
        
        score = np.log(self.pdf(Y)).sum()
        return score


class NormalModel(BaseModel):
    
    '''
    i.i.d. normal distribution model
    '''
    
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
    
    @property
    def loc(self):
        return self.mu
    
    @property
    def scale(self):
        return self.sigma
    
        
    def fit(self, Y, weights=None):
        
        '''
        Fits the model parameters to an observation sequence.
        weights are optional.
        '''
        
        # prepare
        Y = np.array(Y)
        if weights is None:
            weights = np.ones(Y.shape)
        else:
            weights = np.array(weights)
        
        # estimate mean
        mean = np.average(Y, weights=weights)
        
        # estimate variance
        errors = (Y-mean)**2
        variance = np.average(errors, weights=weights)
        
        # update
        self.mu = mean
        self.sigma = np.sqrt(variance)
        
    def pdf(self, Y):
        
        '''
        Returns the likelihood of each observation in an observation sequence.
        '''
        
        Y = np.array(Y)
        pdf = norm(loc=self.mu, scale=self.sigma).pdf(Y)
        return pdf