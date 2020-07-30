import pandas as pd
import numpy as np
import warnings

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from hmmlearn.hmm import GaussianHMM
from src.dists import GaussianMixtureDistribution
from src.markov import MarkovChain


class HMM():
    def __init__(self, switch_var=True, switch_const=True, k=2):
        
        '''
        
        '''
        
        self.switch_var = switch_var
        self.switch_const = switch_const
        self.k = k
        
        self.params_ = None
        self.se_ = None
        self.tstats_ = None

        self.metrics_ = None
        self.smooth_prob_ = None
        self.filt_prob_ = None
    
    
    def fit(self, y, package='statsmodels', start_params=None, iter=100, **kwargs):
        
        '''
        Fits the Gaussian HMM to the series y.
        '''
        
        assert package in ['statsmodels', 'hmmlearn'], 'package unknown'
        
        if package=='statsmodels':
            #if start_params is None:
                #start_params = np.random.randn(self.k+self.k**2)*0.01
                #m = y.mean()
                #s = y.std()
                #v = y.var()
                #start_params = np.full(self.**2-self.k, 1/self.k).tolist()\
                #                        +(np.random.randn(self.k)*s/2+m).tolist()\
                #                        +(np.random.randn(self.k)*v+s).tolist()
            model = MarkovRegression(endog=y, switching_variance=self.switch_var, switching_trend=self.switch_const, k_regimes=self.k)\
                                .fit(start_params=start_params, maxiter=iter, **kwargs)
            self.params_ = model.params
            self.se_ = model.bse
            self.tstats_ = model.tvalues
            self.metrics_ = pd.Series({'llf': model.llf, 'aic': model.aic, 'bic': model.bic,})
            self.smooth_prob_ = model.smoothed_marginal_probabilities
            self.filt_prob_ = model.filtered_marginal_probabilities
        
        if package=='hmmlearn':

            assert self.switch_var is True and self.switch_const is True, 'only implemented for fully parametrised components'
            t_index = y.index
            y = np.expand_dims(y.values, axis=1)
            model = GaussianHMM(n_components=self.k, n_iter=iter, **kwargs).fit(y)
            trans_probas = model.transmat_.T.reshape(self.k**2,1)[:self.k**2-self.k]
            states = np.arange(self.k)
            p_index=[f'p[{j}->{i}]' for i in states[:-1] for j in states]\
                        +[f'const[{i}]' for i in states]\
                        +[f'sigma2[{i}]' for i in states]
            self.params_ = pd.Series(np.concatenate((trans_probas, model.means_, model.covars_.squeeze(axis=1))).squeeze(), index=p_index)
            llf = model.score(y)
            self.metrics_ = pd.Series({'llf': llf,
                                       'aic': 2*len(self.params_)-2*llf,
                                       'bic': len(self.params_)*np.log(len(y))-2*llf})
            self.smooth_prob_ = pd.DataFrame(model.predict_proba(y), index=t_index)

        return self
    
    
    @property
    def estimates_(self):
        estimates = pd.DataFrame({'estimate': self.params_,
                                  's.e.': self.se_,
                                  't-stat': self.tstats_})
        return estimates
    

    @property
    def transition_matrix_(self):
        k = self.k
        trans_matrix = np.matrix(self.params_[:k**2-k].values.reshape(k-1, k).T)
        trans_matrix = np.append(trans_matrix, 1-trans_matrix.sum(axis=1), axis=1)
        return trans_matrix


    @property
    def steady_state_(self):
        mc = MarkovChain(transition_matrix=self.transition_matrix_)
        steady = mc.steady_state_probabilities
        return steady


    def get_mixture_distribution(self, state='steady_state'):
        if state == 'steady_state':
            probas = self.steady_state_
        elif state == 'latest':
            probas = self.filt_prob_.iloc[-1]
        else:
            assert len(state) == self.k, 'wrong number of state probabilities'
            probas = state

        components = [(self.params_[f'const[{i}]'], self.params_[f'sigma2[{i}]']**0.5, probas[i]) for i in range(self.k)]
        mix = GaussianMixtureDistribution(components=components)
        return mix


    def filtered_moments(self):
        filt_mom = pd.DataFrame(index=self.filt_prob_.index, columns=['mean','var','skew','kurt','entropy'])
        for date, probas in self.filt_prob_.iterrows():
            mix = self.get_mixture_distribution(state=probas.values)
            filt_mom.loc[date] = [*mix.mvsk(), mix.entropy()]

        return filt_mom


    def smoothened_moments(self):
        smooth_mom = pd.DataFrame(index=self.smooth_prob_.index, columns=['mean','var','skew','kurt','entropy'])
        for date, probas in self.smooth_prob_.iterrows():
            mix = self.get_mixture_distribution(state=probas.values)
            smooth_mom.loc[date] = [*mix.mvsk(), mix.entropy()]
        return smooth_mom
        
