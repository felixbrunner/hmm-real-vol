import numpy as np

class MarkovChain:
    
    '''
    A MarkovChain
    '''
    
    def __init__(self, transition_matrix=None, state_vector=None):

        self.transition_matrix = transition_matrix
        self.state_vector = state_vector

        if transition_matrix is not None:
            self.n_states = self.transition_matrix.shape[0]
        elif self.state_vector is not None:
            self.n_states = len(self.state_vector)


    def set_transition_matrix(self, transition_matrix):
        self.transition_matrix = transition_matrix
        

    def set_state_vector(state_vector):
        self.state_vector = state_vector
    

    @property
    def steady_state_probabilities(self, set_state=False):
        dim = np.array(self.transition_matrix).shape[0]
        q = np.c_[(self.transition_matrix-np.eye(dim)),np.ones(dim)]
        QTQ = np.dot(q, q.T)
        steady_state_probabilities = np.linalg.solve(QTQ,np.ones(dim))
        if set_state:
            self.state_vector = steady_state_probabilities
        return steady_state_probabilities


    def iterate(self, steps=1, return_state_vector=False):
        self.state_vector = np.dot(self.state_vector, np.linalg.matrix_power(self.transition_matrix, steps))
        if return_state_vector:
            return self.state_vector
    

    def rvs(self, T=1):
        draw = np.random.choice(self.n_states, size=1, p=self.state_vector)[0]
        sample = [draw]
        for t in range(1,T):
            draw = np.random.choice(self.n_states, size=1, p=self.transition_matrix[draw])[0]
            sample += [draw]
        if T is 1:
            sample = sample[0]
        return sample
    
    @property
    def expected_durations(self):
        expected_durations = (np.ones(self.n_states)-np.diag(self.transition_matrix))**-1
        return expected_durations


    def entropy(self):
        
        '''
        Calculate Shannon's entropy based on logarithms with base n of the n state probabilities.
        '''
        
        entropy = 0
        for p in self.state_vector:
            if p == 0:
                pass
            else:
                entropy += p*np.log(p)/np.log(self.n_states)
        return abs(entropy)