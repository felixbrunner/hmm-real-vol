import pandas as pd
import numpy as np
import warnings

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from hmmlearn.hmm import GaussianHMM

from src.dists import GaussianMixtureDistribution
from src.markov import MarkovChain


class HiddenMarkovModel(MixtureModel, MarkovChain):
    
    '''
    Hidden Markov Model class
    '''
    
    def __init__(self, emission_models=None, transition_matrix=None, state_vector=None):
        self.transition_matrix = transition_matrix
        self.state_vector = state_vector
        self.emission_models = emission_models
        self.is_fitted = False
        
        
    @property
    def emission_models(self):
        
        '''
        A tuple of emission models associated with the Markov states.
        '''
        
        return self._emission_models
    
    @emission_models.setter
    def emission_models(self, emission_models):
        if emission_models is not None:
            emission_models = tuple(emission_models)
            if self.transition_matrix is not None:
                assert len(emission_models) == self.transition_matrix.shape[0], \
                    'number of emission models inconsitent'
            elif self.state_vector is not None:
                assert len(emission_models) == self.state_vector.shape[1], \
                    'number of emission models inconsitent'
            self._emission_models = emission_models
            
        else:
            self._emission_models = None
            
            
    @property
    def components(self):
        
        '''
        The mixture distribution components.
        '''
        
        weights = self.state_vector.squeeze()
        components = [(component, float(weight)) for (component, weight) in zip(self.emission_models, weights)]
        return components
        
    
    def fit(self, Y, method='baumwelch', max_iter=100, threshold=1e-6):
        
        '''
        Fits the model to a sample of data.
        '''
        
        if method == 'baumwelch':
            self = self._estimate_baum_welch(Y, max_iter=max_iter, threshold=threshold, return_fit=False)
        else:
            raise NotImplementedError('fitting algorithm not implemented')
        
        self.is_fitted = True
    
    
    def _estimate_baum_welch(self, Y, max_iter=100, threshold=1e-6, return_fit=False):
        
        '''
        Performs parameter estimation with the Baum-Welch algorithm.
        Returns a fitted model.
        Returns the the fitted model and parameters of the estimation if return_fit=True.
        '''
        
        # initialise
        Y = np.array(Y)
        A_, pi_, models_ = self._initialise_baum_welch()
        self._check_baum_welch_inputs(A_, pi_, models_)
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
                converged = True
                break
        else:
            converged = False
            warnings.warn('maximum number of iterations reached')
                
        self._update_attributes(A_, models_, Alpha)
        
        if return_fit:
            fit = {'converged': converged,
                   'iterations': iteration,
                   'scores': scores,
                   'pdfs': B_,
                   'smoothened_probabilities': Gamma,
                   'filtered_probabilities': Alpha}
            
            return self, fit
        else:
            return self
    
    
    def _initialise_baum_welch(self):
        
        '''
        Returns initial values for the Baum-Welch algorithm.
        Part of Baum-Welch algorithm.
        '''
        
        assert self.emission_models is not None, \
            'emission models not specified'
            
        if self.state_vector is None:
            self.steady_state(set_state=True)
        if self.transition_matrix is None:
            self.transition_matrix = np.full([self.n_states, self.n_states], 1/self.n_states)
        
        A = self.transition_matrix
        models = self.emission_models
        pi = self.state_vector
        return A, pi, models
    
    
    def _check_baum_welch_inputs(self, A, pi, models):
        
        '''
        Checks the dimension match of algorithm inputs.
        Part of Baum-Welch algorithm.
        '''

        assert len(models) == A.shape[0] == A.shape[1] == pi.shape[1], \
            'dimension mismatch'
    
    
    def _score(self, Y, emission_models, Gamma):
        
        '''
        Returns the overall model score and component model pdf values for each observation.
        Part of Baum-Welch algorithm.
        '''
        
        B = self._evaluate_emission_models(Y, emission_models)
        score = np.log((B * Gamma).sum(axis=1)).sum(axis=0)
        return score, B
    
    
    def _evaluate_emission_models(self, Y, emission_models):
        
        '''
        Returns component model pdf values for each observation.
        Part of Baum-Welch algorithm.
        '''
        
        B = np.concatenate([model.pdf(Y).reshape(-1, 1) for model in emission_models], axis=1)
        return B
    
    
    def _do_e_step(self, Y, A, B, pi):
        
        '''
        Performs all steps of the E-step and returns temporary variables.
        All data state probabilities are updated based on the existing component models.
        Part of Baum-Welch algorithm.
        '''
        
        Alpha, C = self._forward_pass(A, B, pi)
        Beta = self._backward_pass(A, B, pi, C)
        Gamma = self._emission_odds(Alpha, Beta)
        Xi = self._transition_odds(A, B, Alpha, Beta)
        return Alpha, Gamma, Xi
    
    
    def _forward_pass(self, A, B, pi):
        
        '''
        Returns filtered probabilities of the data together with each scaling factor.
        Part of Baum-Welch algorithm.
        '''
        
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
        
        '''
        Returns smoothened probabilities of the data.
        Part of Baum-Welch algorithm.
        '''
        
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
        
        '''
        Returns odds for each observation to be emitted by each component model.
        Part of Baum-Welch algorithm.
        '''
        
        total = Alpha * Beta
        Gamma = total/total.sum(axis=1).reshape(-1, 1)
        return Gamma
    
    
    def _transition_odds(self, A, B, Alpha, Beta):
        
        '''
        Returns the odds of each state to transition from each state to each state.
        Part of Baum-Welch algorithm.
        '''
        
        Alpha_block = np.kron(Alpha[:-1], np.ones(A.shape[0]))
        B_Beta_block = np.kron(np.ones(A.shape[0]), B[1:]*Beta[1:])
        total = Alpha_block * B_Beta_block * A.reshape(1, -1)
        Xi = total/total.sum(axis=1).reshape(-1, 1)
        return Xi
    
    
    def _do_m_step(self, Y, models, Gamma, Xi):
        
        '''
        Performs all steps of the M-step and returns temporary variables.
        All component models are reestimated and parameters updated.
        Part of Baum-Welch algorithm.
        '''
        
        A_ = self._update_transition_matrix(Gamma, Xi)
        models_ = self._update_model_parameters(Y, models, Gamma)
        pi_ = self._update_initial_state(Gamma)
        return A_, models_, pi_
    

    def _update_transition_matrix(self, Gamma, Xi):
        
        '''
        Returns an updated Markov transition matrix.
        Part of Baum-Welch algorithm.
        '''
        
        numerator = Xi.sum(axis=0)
        denominator = np.kron(Gamma[:-1], np.ones(Gamma.shape[1])).sum(axis=0)
        A_ = (numerator/denominator).reshape(Gamma.shape[1], Gamma.shape[1])
        return A_
    
    
    def _update_model_parameters(self, Y, emission_models, Gamma):
        
        '''
        Returns updated emission models.
        Part of Baum-Welch algorithm.
        '''
        
        models_ = []
        for model, weights in zip(emission_models, Gamma.T):
            model.fit(Y, weights)
            models_ += [model]
        return tuple(models_)
    
    
    def _update_initial_state(self, Gamma):
        
        '''
        Returns updated initial state probabilities.
        Part of Baum-Welch algorithm.
        '''
        
        return Gamma[0].reshape(1, -1)
    
    
    def _update_attributes(self, A_, models_, Alpha):
        
        '''
        Updates the HMM attributes in place.
        Part of Baum-Welch algorithm.
        '''
        
        # ensure total transition probabilities are 1
        if (A_.sum(axis=1) != 1).any():
            A_ = A_.round(6)/A_.round(6).sum(axis=1)
            warnings.warn('Transition matrix rounded to 6 decimal places')
        self.transition_matrix = A_
        
        self.emission_models = models_
        
        state_vector = Alpha[-1]
        # ensure total state probability is 1
        if state_vector.sum() != 1:
            state_vector = state_vector.round(8)/state_vector.round(8).sum()
            warnings.warn('State vector rounded to 8 decimal places')
        self.state_vector = state_vector
    
    
    @property
    def distribution(self):
        
        '''
        Extracts and returns a MixtureDistribution object
        with the current state vector as weights.
        '''
        
        mix = MixtureDistribution(components=self.components)
        return mix
    
    
    @property
    def mixture_distribution(self):
        
        '''
        Extracts and returns a MixtureDistribution object
        with the current state vector as weights.
        '''
        
        return self.distribution
    
    @property
    def markov_chain(self):
        
        '''
        Extracts and returns a MarkovChain object
        with the transition matrix and state vector as parameters.
        '''
                
        mc = MarkovChain(transition_matrix=self.transition_matrix, state_vector=self.state_vector)
        return mc