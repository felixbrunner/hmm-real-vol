import pandas as pd
import numpy as np
import warnings

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from hmmlearn.hmm import GaussianHMM
from src.dists import GaussianMixtureDistribution
from src.markov import MarkovChain


class HMM():
    def __init__(self, emission_models=(), transition_matrix=None, start_probas=None, switch_var=True, switch_const=True, k=None):
        
        '''
        
        '''

        self.emission_models = emission_models
        self.transition_matrix = transition_matrix
        self.start_probas = start_probas
        
        self.switch_var = switch_var
        self.switch_const = switch_const
        self.k = k
        
        self.params_ = None
        self.se_ = None
        self.tstats_ = None

        self.metrics_ = None
        self.smooth_prob_ = None
        self.filt_prob_ = None
    
    
    def fit(self, y, package='baumwelch', start_params=None, iter=100, **kwargs):
        
        '''
        Fits the Gaussian HMM to the series y.
        '''
        
        assert package in ['statsmodels', 'hmmlearn', 'baumwelch'], 'package unknown'
        
        if package == 'statsmodels':
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

        if package == 'baumwelch':
            self = self._estimate_baum_welch(np.array(y), max_iter=iter, **kwargs)

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
        

    def _check(self, A, pi, models):
        '''OK'''
        assert len(models) == A.shape[0] == A.shape[1] == pi.shape[1], 'dimension mismatch'
    
    @property
    def steady_state(self):
        '''FIX'''
        k = self.transition_matrix.shape[0]
        steady_state = np.full(k, 1/k).reshape(1, -1) @ self.transition_matrix
        return steady_state
        
    def _initialise_baum_welch(self, Y):
        '''FIX'''
        if self.start_probas is None:
            self.start_probas = self.steady_state
        
        A = np.array(self.transition_matrix)
        models = self.emission_models
        pi = np.array(self.start_probas).reshape(1, -1)
        return A, pi, models
    
    def _evaluate_emission_models(self, Y, emission_models):
        '''OK'''
        B = np.concatenate([model.pdf(Y).reshape(-1, 1) for model in emission_models], axis=1)
        return B
        
    def _forward_pass(self, A, B, pi):
        '''OK'''
        # initialise forward pass with first observation
        alpha_0 = pi * B[0]
        c_0 = 1/alpha_0.sum()
        
        # save values & scaling factor
        Alpha = alpha_0*c_0
        C = [c_0]
        
        # iterate
        for b_t in B[1:]:
            # calculate
            alpha_t = (b_t * Alpha[-1] @ A).reshape(1, -1)
            c_t = 1/alpha_t.sum()
            
            # save
            Alpha = np.concatenate((Alpha, alpha_t*c_t), axis=0)
            C += [c_t]
            
        C = np.array(C).reshape(-1, 1)
        return Alpha, C
            
    def _backward_pass(self, A, B, pi, C):
        '''OK'''
        # initialise backward pass as one
        beta_T = np.ones(pi.shape)
        
        # save values & scaling factor
        Beta = beta_T*C[-1]
        
        # iterate
        for b_t, c_t in zip(B[:0:-1],C[len(C)-2::-1]):
            # calculate
            beta_t = (b_t * Beta[0] @ A.T).reshape(1, -1)
            
            # save
            Beta = np.concatenate((beta_t*c_t, Beta), axis=0)
            
        return Beta
    
    def _emission_odds(self, Alpha, Beta):
        '''OK'''
        total = Alpha * Beta
        Gamma = total/total.sum(axis=1).reshape(-1, 1)
        return Gamma
    
    def _transition_odds(self, A, B, Alpha, Beta):
        '''OK'''
        Alpha_block = np.kron(Alpha[:-1], np.ones(A.shape[0]))
        B_Beta_block = np.kron(np.ones(A.shape[0]), B[1:]*Beta[1:])
        total = Alpha_block * B_Beta_block * A.reshape(1, -1)
        Xi = total/total.sum(axis=1).reshape(-1, 1)
        return Xi
        
    def _do_e_step(self, Y, A, B, pi):
        '''OK'''
        Alpha, C = self._forward_pass(A, B, pi)
        Beta = self._backward_pass(A, B, pi, C)
        Gamma = self._emission_odds(Alpha, Beta)
        Xi = self._transition_odds(A, B, Alpha, Beta)
        return Alpha, Gamma, Xi
    
    def _update_transition_matrix(self, Gamma, Xi):
        '''OK'''
        numerator = Xi.sum(axis=0)
        denominator = np.kron(Gamma[:-1], np.ones(Gamma.shape[1])).sum(axis=0)
        A_ = (numerator/denominator).reshape(Gamma.shape[1], Gamma.shape[1])
        return A_
    
    def _update_parameters(self, Y, emission_models, Gamma):
        '''OK'''
        models_ = []
        for model, weights in zip(emission_models, Gamma.T):
            model.fit(Y, weights)
            models_ += [model]
        return tuple(models_)
    
    def _update_initial_state(self, Gamma):
        '''OK'''
        return Gamma[0].reshape(1, -1)
    
    def _do_m_step(self, Y, models, Gamma, Xi):
        '''OK'''
        A_ = self._update_transition_matrix(Gamma, Xi)
        models_ = self._update_parameters(Y, models, Gamma)
        pi_ = self._update_initial_state(Gamma)
        return A_, models_, pi_
    
    def _score(self, Y, emission_models, Gamma):
        '''OK'''
        B = self._evaluate_emission_models(Y, emission_models)
        score = np.log((B * Gamma).sum(axis=1)).sum(axis=0)
        return score, B
    
    def _update_attributes(self, A_, models_, pi_, Gamma, Alpha):
        '''OK'''
        self.transition_matrix = A_
        self.emission_models = models_
        self.start_probas = pi_
        self.smooth_prob_ = Gamma
        self.filt_prob_ = Alpha
    
    def _estimate_baum_welch(self, Y, max_iter=100, threshold=1e-6):
        '''OK'''
        # initialise
        A_, pi_, models_ = self._initialise_baum_welch(Y)
        self._check(A_, pi_, models_)
        score_, B_ = self._score(Y, models_, pi_)
        
        # store
        iteration = 0
        scores = {iteration: score_}
        
        while iteration < max_iter:
            iteration += 1
            Alpha, Gamma, Xi = self._do_e_step(Y, A_, B_, pi_)            
            A_, models_, pi_ = self._do_m_step(Y, models_, Gamma, Xi)
            score_, B_ = self._score(Y, models_, Gamma)
            scores[iteration] = score_
            
            if abs(scores[iteration]-scores[iteration-1]) < threshold:
                break
        else:
            warnings.warn('maximum number of iterations reached')
                
        self._update_attributes(A_, models_, pi_, Gamma, Alpha)
        self.convergence_ = scores
        
        return self
            
    # def fit(self, Y, method='baumwelch', **kwargs):
    #     '''OK'''
    #     assert method in ['baumwelch'], 'method unknown'
        
    #     if method == 'baumwelch':
    #         self = self._estimate_baum_welch(Y, **kwargs)