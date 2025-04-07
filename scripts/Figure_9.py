#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 17:51:01 2025

@author: kbhaakman
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as mticker
from shared_functions import load_PSMSL_data, correct_surge, define_model
plt.style.use('ggplot')
rcParams.update({'font.size': 12})


### OPTIONS ###
t0 = 1890 #inclusive
t1 = 2022 #exclusive
correct_surge_GTSM = True
r_vec = np.logspace(0, 12, 40)
################



stations = ['Vlissingen', 'HvH', 'IJmuiden', 'DenHelder', 'Harlingen', 'Delfzijl']
title_names = ['Vlissingen', 'Hoek van Holland', 'IJmuiden', 'Den Helder', 'Harlingen', 'Delfzijl']
fn_base = '../data/PSMSL/{}_yearly.rlrdata'
filenames = [fn_base.format(station) for station in stations]




def compute_trends_before_after(fn):
    station = fn.split('/')[-1].split('_')[0]
    t, y = load_PSMSL_data(fn, t0, t1)
    if correct_surge_GTSM:
        y = correct_surge(y, station)
        
    model = define_model(t, y)
    res = model.fit(disp=False)
    #res.plot_diagnostics()
    print(res.summary())

    sigma_xi = np.sqrt(res.params[1]) # fitted AR(1) noise variance
    phi = res.params[2] # fitted AR(1) parameter
    sigma_eps = np.sqrt((1 - phi**2) * sigma_xi**2) # observation noise variance
    r_hat = sigma_eps**2 / res.params[0] # fitted noise variance ratio

    
    mean_trends_before_1993 = np.zeros(len(r_vec))
    mean_trends_after_1993 = np.zeros(len(r_vec))
    mean_trends_before_1993_variance = np.zeros(len(r_vec))
    mean_trends_after_1993_variance = np.zeros(len(r_vec))
    idx_1993 = np.where(t == 1993)[0][0]
    
    for i in range(len(r_vec)):
        sigma_eta = sigma_eps / np.sqrt(r_vec[i])
        constraints = {}
        constraints['sigma2.ar'] = sigma_xi**2
        constraints['ar.L1'] = phi
        constraints['sigma2.trend'] = sigma_eta**2
        res = model.fit_constrained(constraints, disp=False)

        trend_instantaneous = res.states.smoothed[:,1] #extract smoothed trend variable (nu, not mu, in Durbin and Koopman notation)
    

        mean_trends_before_1993[i] = 10 * np.mean(trend_instantaneous[:idx_1993]) #factor 10 to to go from cm to mm
        mean_trends_before_1993_variance[i] = 100 * compute_uncertainty_in_mean_trend(res, idx0=0, idx1=idx_1993) #factor 100 is to go from cm^2 to mm^2
        mean_trends_after_1993[i] = 10 * np.mean(trend_instantaneous[idx_1993:])
        mean_trends_after_1993_variance[i] = 100 * compute_uncertainty_in_mean_trend(res, idx0=idx_1993, idx1=len(t))
    
    return mean_trends_before_1993, mean_trends_after_1993, mean_trends_before_1993_variance, mean_trends_after_1993_variance, r_hat


def compute_uncertainty_in_mean_trend(res, idx0, idx1):
    """
    This function computes the variance of the mean trend over several timesteps.
    Since the trend states at different timesteps are not independent, 
    the autocovariance of the states at different timesteps is accounted for.
    
    This computations follows the following formula:
    var(mean(nu_k)) = 1/N^2 * var(sum(nu_k)) 
                    = 1/N^2 * (sum(var(nu_k)) + 2 * sum(cov(nu_k, nu_{k+h})))

    Parameters
    ----------
    res : object
        UnobservedComponentsResults object from statsmodels.
    idx0 : int
        first index of trend average (inclusive)
    idx1 : int
        last index of trend average (exclusive)

    Returns
    -------
    var_mean_nu_k : float
        variance of the mean of trend states

    """
    N = idx1 - idx0 #number of timesteps over which mean trend is computed
    trend_variances = res.smoothed_state_cov[1,1,:] #lag 0 variances of instantaneous trend states
    term1 = np.sum(trend_variances[idx0:idx1])
    
    term2 = 0
    for k in range(N): #loop over timesteos
        for h in range(1,N-k): #loop over lags 
            covariance = res.smoother_results.smoothed_state_autocovariance(start=idx0+k, end=idx0+k+1, lag=-h)
            term2 += covariance[1,1,0] #add covariance between nu_k and nu_{k+h}
            
    var_sum_nu_k = term1 + 2 * term2 #sum of variances + 2 times off-diagonal terms
    var_mean_nu_k = 1 / N**2 * var_sum_nu_k 
    
    return var_mean_nu_k

def plot_trend_with_confidence_intervals(ax, mean, variance, station, label):
    lower = mean[station,:] - 1.96 * np.sqrt(variance[station,:])
    upper = mean[station,:] + 1.96 * np.sqrt(variance[station,:])
    ax.plot(r_vec, mean[station,:], label=label)
    ax.fill_between(r_vec, lower, upper, alpha=0.3)
    return

def plot_mean_trends_six(before, after, before_variance, after_variance, r_hats):
    
    before_label = '1890-1993'
    after_label = '1993-2022'
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12,6))
    #fig.tight_layout() 
    ax = axes[0,0]
    station = 0
    plot_trend_with_confidence_intervals(ax, after, after_variance, station, after_label)
    plot_trend_with_confidence_intervals(ax, before, before_variance, station, before_label)
    ax.axvline(r_hats[station], color='k', linestyle='dashed', label='MLE')
    ax.set_xscale('log')
    ax.set_ylabel('Sea level trend [mm/yr]', fontsize=12)
    ax.set_title(title_names[station])
    
    ax = axes[0,1]
    station = 1
    plot_trend_with_confidence_intervals(ax, after, after_variance, station, after_label)
    plot_trend_with_confidence_intervals(ax, before, before_variance, station, before_label)
    ax.axvline(r_hats[station], color='k', linestyle='dashed', label='MLE')
    ax.set_xscale('log')
    ax.set_title(title_names[station])
    
    ax = axes[0,2]
    station = 2
    plot_trend_with_confidence_intervals(ax, after, after_variance, station, after_label)
    plot_trend_with_confidence_intervals(ax, before, before_variance, station, before_label)
    ax.axvline(r_hats[station], color='k', linestyle='dashed', label='MLE')
    ax.set_xscale('log')
    ax.set_title(title_names[station])
    
    ax = axes[1,0]
    station = 3
    plot_trend_with_confidence_intervals(ax, after, after_variance, station, after_label)
    plot_trend_with_confidence_intervals(ax, before, before_variance, station, before_label)
    ax.axvline(r_hats[station], color='k', linestyle='dashed', label='MLE')
    ax.set_xscale('log')
    ax.set_ylabel('Sea level trend [mm/yr]', fontsize=12)
    ax.set_xlabel(r'NVR [-]')
    ax.set_title(title_names[station])
    
    #ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    
    ax = axes[1,1]
    station = 4
    plot_trend_with_confidence_intervals(ax, after, after_variance, station, after_label)
    plot_trend_with_confidence_intervals(ax, before, before_variance, station, before_label)
    ax.axvline(r_hats[station], color='k', linestyle='dashed', label='MLE')
    ax.set_xscale('log')
    ax.set_xlabel(r'NVR [-]')
    ax.set_title(title_names[station])
    
    ax = axes[1,2]
    station = 5
    plot_trend_with_confidence_intervals(ax, after, after_variance, station, after_label)
    plot_trend_with_confidence_intervals(ax, before, before_variance, station, before_label)
    ax.axvline(r_hats[station], color='k', linestyle='dashed', label='MLE estimate')
    ax.set_xscale('log')
    ax.legend()
    ax.set_xlabel(r'NVR [-]')
    ax.set_title(title_names[station])
    fig.tight_layout()
    plt.savefig('../figures/Figure_9.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    
    return

def main():
   
    mean_trends_before = np.zeros((len(filenames), len(r_vec)))
    mean_trends_after = np.zeros_like(mean_trends_before)
    mean_trends_before_variance = np.zeros_like(mean_trends_before)
    mean_trends_after_variance = np.zeros_like(mean_trends_before)
    r_hats = np.zeros(len(filenames))
    
    for i, fn in enumerate(filenames):
        mean_trends_before[i,:], mean_trends_after[i,:], mean_trends_before_variance[i,:], mean_trends_after_variance[i,:], r_hats[i] = compute_trends_before_after(fn)
        
    plot_mean_trends_six(mean_trends_before, mean_trends_after, mean_trends_before_variance, mean_trends_after_variance, r_hats)
    
    return 

if __name__ == '__main__':
    main()