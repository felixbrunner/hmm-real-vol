import numpy as np
import scipy as sp
import pandas as pd

def calculate_kolmogorov_smirnov_distance(mu0, mu1, sigma0, sigma1):
    if np.isnan(mu0) or np.isnan(mu1) or np.isnan(sigma0) or np.isnan(sigma1):
        KS = np.nan
        
    else:    
        a = 1/(2*sigma0**2) - 1/(2*sigma1**2)
        b = mu1/sigma1**2 - mu0/sigma0**2
        c = mu0**2/(2*sigma0**2) - mu1**2/(2*sigma1**2) - np.log(sigma1/sigma0)
    
        if a == 0:
            if b == 0:
                KS = 0
            else:
                x_sup = -c/b
                KS = abs(sp.stats.norm.cdf(x_sup, mu0, sigma0) - sp.stats.norm.cdf(x_sup, mu1, sigma1))
        else:
            x_1 = (-b + (b**2-4*a*c)**0.5) / (2*a)
            x_2 = (-b - (b**2-4*a*c)**0.5) / (2*a)
        
            KS_1 = abs(sp.stats.norm.cdf(x_1, mu0, sigma0) - sp.stats.norm.cdf(x_1, mu1, sigma1))
            KS_2 = abs(sp.stats.norm.cdf(x_2, mu0, sigma0) - sp.stats.norm.cdf(x_2, mu1, sigma1))
        
            KS = max(KS_1,KS_2)
    
    return KS

def normal_central_moment(sigma, moment):

    '''Central moments of a normal distribution with any mean.
    Note that the first input is a standard deviation, not a variance.'''

    if moment % 2 == 1:
        #odd moments of a normal are zero
        normal_moment = 0 
    else:
        #even moments are given by sigma^n times the double factorial
        normal_moment = sigma**moment * sp.special.factorialk(moment-1, 2) 
    return normal_moment


def calculate_steady_state_probabilities(transition_matrix):
    dim = np.array(transition_matrix).shape[0]
    q = np.c_[(transition_matrix-np.eye(dim)),np.ones(dim)]
    QTQ = np.dot(q, q.T)
    steady_state_probabilities = np.linalg.solve(QTQ,np.ones(dim))
    return steady_state_probabilities


def iterate_markov_chain(state_probabilities, transition_matrix, steps):
    new_state = np.dot(state_probabilities, np.linalg.matrix_power(transition_matrix, steps))
    return new_state


def calculate_binary_entropy(probability):
    entropy = -1 * (probability*np.log2(probability) + (1-probability)*np.log2(1-probability))
    return abs(entropy)


def calculate_columnwise_autocorrelation(df, lag=1):
    autocorr = pd.Series(index=df.columns)
    for column in df.columns:
        autocorr[column] = df[column].astype('float64').autocorr(lag=lag)
    return autocorr


def export_df_to_latex(df, filename='file.tex', **kwargs):
    if filename[-4:] != '.tex':
        filename += '.tex'
    df.to_latex(buf=filename, multirow=False, multicolumn_format ='c', na_rep='', escape=False, **kwargs)
    
    
def calculate_shannon_entropy(vector):
    '''
    calculate the shannon entropy measure from a vector of probabilities
    '''
    
    n = len(vector)
    entropy = 0
    for v in vector:
        if v == 0:
            pass
        else:
            entropy += v*np.log(v)/np.log(n)
    return abs(entropy)


def shrink_outliers(Series, alpha=1.96, lamb=1):
    '''
    This function shrinks outliers in a series towards the threshold values.
    The parameter alpha defines the threshold values as a multiple of one sample standard deviation.
    The parameter lamb defines the degree of shrinkage of outliers towards the thresholds.
    
    The transformation is as follows:
    if the z score is inside the thresholds f(x)=x
    if it is above the upper threshold f(x)=1+1/lamb*ln(x+(1-lamb)/lamb)-1/lamb*ln(1/lamb)
    if it is below the lower threshold f(x)=-1-1/lamb*ln(-x+(1-lamb)/lamb)+1/lamb*ln(1/lamb)
    '''
    
    z_scores = (Series-Series.mean())/Series.std()
    adjusted_scores = z_scores/alpha
    adjusted_scores[adjusted_scores.values>1] = 1+1/lamb*np.log(adjusted_scores[adjusted_scores.values>1]+(1-lamb)/lamb)-1/lamb*np.log(1/lamb)
    adjusted_scores[adjusted_scores.values<-1] = -1-1/lamb*np.log((1-lamb)/lamb-adjusted_scores[adjusted_scores.values<-1])+1/lamb*np.log(1/lamb)
    new_z_scores = adjusted_scores*alpha
    new_series = new_z_scores*Series.std()+Series.mean()
    return new_series


def get_unique_values_from_list_of_lists(input_list):
    full_list = [j for i in input_list for j in i]
    short_list = list(dict.fromkeys(full_list))
    return short_list


def standardise_dataframe(df, ax=0):
    df = df.subtract(df.mean(axis=ax), axis=ax)
    df = df.divide(df.std(axis=ax), axis=ax)
    return df


def total_return_from_returns(returns): #returns total return of a return series
    return (returns + 1).prod(skipna=False) - 1


def fill_inside_na(df, meth = 'zero', lim = None): #fills missing values in the middle of a series (but not at beginning and end)
    nans = pd.DataFrame(df.bfill().isna() | df.ffill().isna())
    if meth is 'zero':
        df = df.fillna(0)
    else:
        df = df.fillna(method = meth, limit = lim)
    df[nans] = np.nan
    return df


def realised_volatility_from_returns(returns):
    return ((returns**2).mean(skipna=False))**0.5