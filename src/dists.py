### IMPORTS ###

import numpy as np
import scipy as sp
import pandas as pd

#import scipy.stats as ss
#from src.utils import normal_central_moment

class BaseDistribution:
    
    '''
    Base class for distributions.
    '''
    
    def __init__(self):
        pass
    

    def std(self):

        '''
        Returns the distribution standard deviation.
        '''

        return self.var()**0.5


    def exkurt(self):

        '''
        Returns the excess kurtosis.
        '''

        return self.kurt()-3


    def mvsk(self):
    
        '''
        Returns the first four standardised moments about the mean.
        '''
    
        m = self.mean()
        v = self.var()
        s = self.skew()
        k = self.kurt()
        return (m, v, s, k)



class NormalDistribution(BaseDistribution):
    
    '''
    A normal distribution.
    If no parameters are specified, a standard normal distribution.
    '''
    
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        
        
    @property
    def mu(self):
        
        '''
        The distribution mean.
        '''
        
        return self._mu
    
    @mu.setter
    def mu(self, mu):
        assert type(mu) == int or type(mu) == float or type(mu) == np.float64, \
            'mu needs to be numeric'
        self._mu = mu
        
        
    @property
    def sigma(self):
        
        '''
        The distribution standard deviation.
        '''
        
        return self._sigma
    
    @sigma.setter
    def sigma(self, sigma):
        assert type(sigma) == int or type(sigma) == float or type(sigma) == np.float64, \
            'sigma needs to be numeric'
        self._sigma = sigma
    
    
    def central_moment(self, moment):

        '''
        Returns the central moments of input order.
        '''
        
        assert moment>0 and type(moment)==int, \
            'moment needs to be a positive integer'

        if moment % 2 == 1:
            #odd moments of a normal are zero
            central_moment = 0 
        else:
            #even moments are given by sigma^n times the double factorial
            central_moment = self.sigma**moment * sp.special.factorialk(moment-1, 2)
        return central_moment
    
    
    def standardised_moment(self, moment):
    
        '''
        Returns the normalised moment of input order.
        '''
        
        assert moment>0 and type(moment)==int, \
            'moment needs to be a positive integer'
        
        central_moment = self.central_moment(moment)
        if (moment<=2):
            standardised_moment = central_moment
        else:
            standardised_moment = central_moment / self.var()**(moment/2)
        return standardised_moment
    
    
    def mean(self):
        
        '''
        Returns the distribution mean.
        '''
        
        return self.mu
    
    
    def var(self):
        
        '''
        Returns the distribution variance.
        '''
        
        var = self.standardised_moment(2)
        return var
    
    
    def skew(self):
        
        '''
        Returns the distribution skewness.
        '''
        
        skew = self.standardised_moment(3)
        return skew
    
    
    def kurt(self):
        
        '''
        Returns the distribution kurtosis.
        '''
        
        kurt = self.standardised_moment(4)
        return kurt
    
    
    def pdf(self, x):
        y = sp.stats.norm(loc=self.mu, scale=self.sigma).pdf(x)
        return y
    
    
    def cdf(self, x):
        y = sp.stats.norm(loc=self.mu, scale=self.sigma).cdf(x)
        return y
    
    
    def rvs(self, size=1):
        sample = sp.stats.norm(loc=self.mu, scale=self.sigma).rvs(size=size)

        if size == 1:
            sample = sample[0]
        return sample


    def __str__(self):

        '''
        Returns a summarizing string
        '''

        string = 'NormalDistribution(mu={}, sigma={})'.format(round(self.mu, 4), round(self.sigma, 4))
        return string


class MixtureDistribution(BaseDistribution):
    
    '''
    A mixture distribution is a list of triples that parametrise the components of a Gaussian mixture distribution.
    Each triple is a tuple of mean, standard deviation and probability weight of the component.
    '''
    
    def __init__(self, components=None):
        self.components = components
        
    
    def _check_component(self, component):
        
        '''
        Checks component inputs.
        '''
        
        dist, weight = component
        assert isinstance(dist, BaseDistribution), \
            'unknown component distribution type'
        assert type(weight) == float or type(weight) == int or type(weight) == np.float64, \
            'weight needs to be numeric'
    
    @property
    def components(self):
        
        '''
        List of (distribution, weight) tuples.
        Distributions need to be instances of Base Distribution.
        Weights need to be numbers.
        '''
        
        return self._components
    
    @components.setter
    def components(self, components):
        assert type(components) == list, \
            'components needs to be a list of tuples'
        for component in components:
            self._check_component(component)
        self._components = components
        
        
    @property
    def distributions(self):
        
        '''
        Returns a list of component distributions.
        '''
        
        distributions = [component[0] for component in self.components]
        return distributions
        
        
    @property
    def weights(self):
        
        '''
        Returns a list of component weights.
        '''
        
        weights = [component[1] for component in self.components]
        return weights
    
    
    @property
    def n_components(self):
        
        '''
        Returns the number of components.
        '''
        
        return len(self.components)
    
    
    def add_component(self, distribution, weight):
        
        '''
        Adds a component to the mixture distribution.
        Inputs needs to be a distribution and a weight.
        '''
        
        component = (distribution, weight)
        self._check_component(component)
        self.components = self.components + [component]

        
    def mean(self):
        
        '''
        Returns the mean.
        '''
        
        mean = sum([component.mean()*weight for (component, weight) in self.components])
        return mean
        
    
    def central_moment(self, moment):

        '''
        Returns the central moment of input order.
        '''

        assert moment > 0, \
            'moment needs to be positive'
    
        if moment is 1:
            return 0
        else:
            mean = self.mean()
            inputs = [(component.mean(), component.std(), weight) for (component, weight) in self.components]
            central_moment = 0
            for (m, s, w) in inputs:
                for k in range(moment+1):
                    product = sp.special.comb(moment, k) * (m-mean)**(moment-k) * sp.stats.norm(loc=0, scale=s).moment(k)
                    central_moment += w * product
            return central_moment
        
        
    def standardised_moment(self, moment):
    
        '''
        Returns the normalised moment of input order.
        '''
    
        if (moment<=2):
            standardised_moment = self.central_moment(moment)
        else:
            variance = self.central_moment(2)
            central_moment = self.central_moment(moment)
            standardised_moment = central_moment / variance**(moment/2)
            if (moment%2==0):
                bias = sp.stats.norm(loc=0, scale=1).moment(moment)
                standardised_moment -= bias
        return standardised_moment
    

    def var(self):
        
        '''
        Returns the distribution variance.
        '''
        
        return self.standardised_moment(2)
    

    def skew(self):
        
        '''
        Returns the distribution skewness.
        '''
        
        return self.standardised_moment(3)
    

    def kurt(self):
        
        '''
        Returns the distribution kurtosis.
        '''
        
        return self.standardised_moment(4)
    
    
    def entropy(self, level='state'):
        
        '''
        Returns Shannon's entropy based on logarithms with base n of the n component probabilities.
        '''
        
        if level == 'state':
            entropy = sp.stats.entropy(self.weights, base=self.n_components)
        else:
            raise NotImplementedError('random variable entropy not implemented')
        return entropy
    
    
    def component_means(self):
        
        '''
        Returns a list of component means.
        '''
        
        means = [distribution.mean() for (distribution, weight) in self.components]
        return means
    
    
    def component_stds(self):
        
        '''
        Returns a list of component standard deviations.
        '''
        
        stds = [distribution.std() for (distribution, weight) in self.components]
        return stds
    
    
    def pdf(self, x):
        
        '''
        Evaluates the probability density function at x.
        '''
        
        y = np.zeros(np.array(x).shape)
        for (component, weight) in self.components:
            y += weight*component.pdf(x)
        return y
    
    
    def cdf(self, x):
        
        '''
        Evaluates the cumulative density function at x.
        '''
        
        y = np.zeros(np.array(x).shape)
        for (component, weight) in self.components:
            y += weight*component.cdf(x)
        return y


    def rvs(self, size=1, return_states=False):
    
        '''
        Draw a random sample from a mixture distribution
        '''
    
        states = np.random.choice(self.n_components, size=size, replace=True, p=self.weights)
        sample = np.fromiter((self.components[i][0].rvs() for i in states), dtype=np.float64)
        
        if size is 1:
            sample = sample[0]
            
        if return_states:
            return (sample, states)
        else:
            return sample


    def __str__(self):

        '''
        Returns a summarizing string
        '''

        string = 'MixtureDistribution(\n'
        for (component, weight) in self.components:
            string += '\t {}, weight={},\n'.format(component.__str__(), weight)
        string += ')'
        return string
    


class ProductDistribution(BaseDistribution):
    
    '''
    A ProducDistribution is a list of tuples that contains the first central moments of the factor distributions.
    Note that the input moments have to be non-standardised and factor draws have to be independent.
    '''
    
    def __init__(self, factors=None):
        self.factors = factors
        
        
    def _check_factor(self, factor):
        
        '''
        Checks factor inputs.
        '''
        
        assert isinstance(factor, BaseDistribution), \
            'unknown factor distribution type'

        
    @property
    def factors(self):
        
        '''
        List of factor distributions.
        Each element needs to be instances of BaseDistribution.
        '''
        
        return self._factors
    
    @factors.setter
    def factors(self, factors):
        assert type(factors) == list, \
            'factors needs to be a list of tuples'
        for factor in factors:
            self._check_factor(factor)
        self._factors = factors
    
    
    @property
    def n_factors(self):
        
        '''
        Returns the number of factors.
        '''
        
        return len(self.factors)
    
    
    def add_factor(self, factor):
        
        '''
        Adds a factor to the mixture distribution.
        Input needs to instance of BaseDistribution.
        '''
        
        self._check_factor(factor)
        self.factors = self.factors + [factor]


    def mean(self):
        
        '''
        Returns the distribution mean.
        '''
        
        prod = 1
        for factor in self.factors:
            m = factor.mean()
            prod *= m
        mean = prod
        return mean
    

    def var(self):
        
        '''
        Returns the distribution variance.
        '''
        
        prod1, prod2 = 1, 1
        for factor in self.factors:
            (m, s) = (factor.mean(), factor.var())
            prod1 *= m**2+s
            prod2 *= m**2
        var = prod1 - prod2
        return var
    

    def skew(self):
        
        '''
        Returns the distribution skewness.
        '''
        
        prod1, prod2, prod3 = 1, 1, 1
        for factor in self.factors:
            (m, s, g) = (factor.mean(), factor.var(), factor.central_moment(3))
            prod1 *= g+3*m*s+m**3
            prod2 *= m*s+m**3
            prod3 *= m**3
        third_central_moment = prod1 - 3*prod2 + 2*prod3
        skew = third_central_moment/(self.var()**1.5)
        return skew

    def kurt(self):
        
        '''
        Returns the distribution kurtosis.
        '''
        
        prod1, prod2, prod3, prod4 = 1, 1, 1, 1
        for factor in self.factors:
            (m, s, g, k) = (factor.mean(), factor.var(), factor.central_moment(3), factor.central_moment(4))
            prod1 *= k+4*m*g+6*m**2*s+m**4
            prod2 *= m*g+3*m**2*s+m**4
            prod3 *= m**2*s+m**4
            prod4 *= m**4
        fourth_central_moment = prod1 - 4*prod2 + 6*prod3 - 3*prod4
        kurt = fourth_central_moment/(self.var()**2)-3
        return kurt


    def rvs(self, size=1):
        
        '''
        Returns a random sample drawn from a mixture distribution
        '''
        
        sample = np.ones(size)
        for factor in self.factors:
            sample *= factor.rvs(size=size)

        if size is 1:
            sample = sample[0]
            
        return sample

    
    def pdf(self):
        raise NotImplementedError('exact pdf unknown')
        
    
    def cdf(self):
        raise NotImplementedError('exact cdf unknown')


    def __str__(self):

        '''
        Returns a summarizing string
        '''

        string = 'ProductDistribution(\n'
        for factor in self.factors:
            string += '\t {},\n'.format(factor.__str__())
        string += ')'
        return string



#### LEGACY CODE ######

# class GaussianMixtureDistribution(BaseDistribution):
    
#     '''
#     A GaussianMixtureDistribution is a list of triples that parametrise the components of a Gaussian mixture distribution.
#     Each triple is a tuple of mean, standard deviation and probability weight of the component.
#     '''
    
#     def __init__(self, components=[]):
#         self.components = components
#         self.n_components = len(self.components)
        
#     def add_component(self,component):
#         self.components += [component]
#         self.n_components += 1
        
#     def central_moment(self, moment):

#         '''
#         Compute the central moments of a mixture of normal components.
#         Moment is the order of the central moment to compute.
#         '''

#         mean = sum([w*m for (m, s, w) in self.components])
    
#         if moment is 1:
#             return mean
#         else:
#             mixture_moment = 0
#             for (m, s, w) in self.components:
#                 for k in range(moment+1):
#                     product = sp.special.comb(moment, k) * (m-mean)**(moment-k) * normal_central_moment(s, k)
#                     mixture_moment += w * product
#             return mixture_moment
        
#     def standardised_moment(self, moment):
    
#         '''
#         Normalised moment of a mixture distribution.
#         '''
    
#         if (moment<=2):
#             mixture_moment = self.central_moment(moment)
#         else:
#             mixture_variance = self.central_moment(2)
#             mixture_central_moment = self.central_moment(moment)
#             mixture_moment = mixture_central_moment / mixture_variance**(moment/2)
#             if (moment%2==0):
#                 bias = normal_central_moment(1,moment)
#                 mixture_moment -= bias
#         return mixture_moment
    

#     def mean(self):
#         return self.standardised_moment(1)
    

#     def var(self):
#         return self.standardised_moment(2)
    

#     def std(self):
#         return self.standardised_moment(2)**0.5
    

#     def skew(self):
#         return self.standardised_moment(3)
    

#     def kurt(self):
        
#         '''
#         Note that the output value is the excess kurtosis.
#         '''
        
#         return self.standardised_moment(4)
    

#     # def mvsk(self):
    
#     #     '''
#     #     The first four standardised moments about the mean of a mixture distribution.
#     #     '''
    
#     #     m = self.mean()
#     #     v = self.var()
#     #     s = self.skew()
#     #     k = self.kurt()
#     #     return (m,v,s,k)

            
#     def rvs(self, sample_size=1):
    
#         '''
#         Draw a random sample from a mixture distribution
#         '''
    
#         weights = [p for (m,s,p) in self.components]
#         norm_params = [(m,s) for (m,s,p) in self.components]
#         draw_from = np.random.choice(self.n_components, size=sample_size, replace=True, p=weights)
#         sample = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in draw_from),dtype=np.float64)
#         if sample_size is 1:
#             sample = sample[0]
#         return sample
    

#     def pdf(self, x):
#         y = np.zeros(np.array(x).shape)
#         for (m, s, w) in self.components:
#             y += w*sp.stats.norm.pdf(x, m, s)
#         return y
    

#     def cdf(self, x):
#         y = np.zeros(np.array(x).shape)
#         for (m, s, w) in self.components:
#             y += w*sp.stats.norm.cdf(x, m, s)
#         return y
    

#     def entropy(self):
        
#         '''
#         Calculate Shannon's entropy based on logarithms with base n of the n component probabilities.
#         '''
        
#         entropy = 0
#         for (m, s, w) in self.components:
#             if w == 0:
#                 pass
#             else:
#                 entropy += w*np.log(w)/np.log(self.n_components)
#         return abs(entropy)
    
#     def get_component_means(self):
#         means = [m for (m,s,w) in self.components]
#         return means
    
#     def get_component_stds(self):
#         stds = [s for (m,s,w) in self.components]
#         return stds
    
#     def get_component_weights(self):
#         weights = [w for (m,s,w) in self.components]
#         return weights