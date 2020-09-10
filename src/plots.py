#from src.dists import GaussianMixtureDistribution
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def plot_pdf(distribution, n_space=1000, std_range=3, ax=None, **kwargs):

    '''
    Plots a MixtureDistribution pdf.
    '''

    # set limits
    lower = distribution.mean()-std_range*distribution.std()
    upper = distribution.mean()+std_range*distribution.std()

    # calculate values
    x = np.linspace(lower, upper, n_space)
    y = distribution.pdf(x)
    
    #plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[16,6])  
    ax.plot(x, y, **kwargs)

    return ax


def plot_mixture_components(mixture_distribution, n_space=1000, std_range=3, ax=None, **kwargs):

    '''
    Plots the weighted component pdfs of a MixtureDistribution.
    '''

    # set limits
    lower = mixture_distribution.mean()-std_range*mixture_distribution.std()
    upper = mixture_distribution.mean()+std_range*mixture_distribution.std()

    # get grid
    x = np.linspace(lower, upper, n_space)
    
    # create plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[16,6])
    
    # calculate & plot
    for (distribution, weight) in mixture_distribution.components:
        y_comp = weight*distribution.pdf(x)
        ax.plot(x, y_comp, linestyle=':', **kwargs)

    return ax


def plot_kernel_density(sample, n_space=1000, std_range=3, ax=None, **kwargs):

    '''
    Plots the kernel density estimate for a sample of data.
    '''

    # set limits
    lower = sample.mean()-std_range*sample.std()
    upper = sample.mean()+std_range*sample.std()

    # create grid
    x = np.linspace(lower, upper, n_space)

    # calculate
    kde = sp.stats.gaussian_kde(sample)
    y = kde(x)
    
    # plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[16,6])
    ax.fill_between(x, y, linewidth=0, alpha=0.3, **kwargs)

    return ax


# def plot_fitted_normal(sample, n_space=1000, std_range=3, ax=None, **kwargs):

#     '''
#     Plots a NormalDistribution pdf.
#     '''

#     # 
#     m = sample.mean()
#     s = sample.std()
#     lower = m - std_range*s
#     upper = m + std_range*s

#     # get grid
#     x = np.linspace(lower, upper, n_space)
#     y = sp.stats.norm.pdf(x, m, s)
    
#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=[16,6])
        
#     ax.plot(x, y, **kwargs)

#     return ax

