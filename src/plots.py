from src.dists import GaussianMixtureDistribution
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def plot_mixture_pdf(mixture_distribution, n_space=1000, std_range=3, ax=None, **kwargs):

    lower = mixture_distribution.mean()-std_range*mixture_distribution.std()
    upper = mixture_distribution.mean()+std_range*mixture_distribution.std()
    x = np.linspace(lower, upper, n_space)
    y = mixture_distribution.pdf(x)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[16,6])
        
    ax.plot(x, y, **kwargs)

    return ax


def plot_mixture_components(mixture_distribution, n_space=1000, std_range=3, ax=None, **kwargs):

    lower = mixture_distribution.mean()-std_range*mixture_distribution.std()
    upper = mixture_distribution.mean()+std_range*mixture_distribution.std()
    x = np.linspace(lower, upper, n_space)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[16,6])
        
    for (m, s, p) in mixture_distribution.components:
        y_comp = p*sp.stats.norm.pdf(x, m, s)
        ax.plot(x, y_comp, linestyle=':', **kwargs)

    return ax


def plot_kernel_density(sample, n_space=1000, std_range=3, ax=None, **kwargs):

    lower = sample.mean()-std_range*sample.std()
    upper = sample.mean()+std_range*sample.std()
    x = np.linspace(lower, upper, n_space)
    kde = sp.stats.gaussian_kde(sample)
    y = kde(x)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[16,6])
        
    ax.fill_between(x, y, linewidth=0, alpha=0.3, **kwargs)

    return ax


def plot_fitted_normal(sample, n_space=1000, std_range=3, ax=None, **kwargs):

    m = sample.mean()
    s = sample.std()
    lower = m - std_range*s
    upper = m + std_range*s
    x = np.linspace(lower, upper, n_space)
    y = sp.stats.norm.pdf(x, m, s)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[16,6])
        
    ax.plot(x, y, **kwargs)

    return ax