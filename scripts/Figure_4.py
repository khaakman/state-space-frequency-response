#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 14:25:43 2025

@author: kbhaakman
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from statsmodels.tsa.statespace.structural import UnobservedComponents


### Settings ###
N_realizations = 100
N_time = 1_000
period = 2*np.pi
r_values = [1e3, 1e6]
###############



def H_RW_amp(omega, omega0, r):
    num = 2 - np.cos(omega0) * 2 * np.cos(omega)
    den = 2 + 4*r + 2*r*np.cos(2*omega0) - (4*r + 1) * np.cos(omega0)*2*np.cos(omega) + r*(2*np.cos(2*omega))
    return num / den


def compute_lombscargle(t, y, normalization='psd'):
    LS = LombScargle(t, y, normalization=normalization)
    frequency, power = LS.autopower(maximum_frequency=0.5, samples_per_peak=20)
    return frequency, np.sqrt(power)


def compute_freq_response(t, r):
    Y = np.zeros((N_realizations, 9990)) 
    X = np.zeros((N_realizations, 9990)) 

    for i in range(N_realizations):
        y = np.random.randn(N_time)
        model = UnobservedComponents(y, level=False, freq_seasonal=[{'period': period, 'harmonics': 1}], stochastic_freq_seasonal=[True], irregular=True)
        
        constraints = {}
        constraints['sigma2.irregular'] = 1
        constraints['sigma2.freq_seasonal_{}(1)'.format(str(period))] = 1 / r
        res = model.fit_constrained(constraints)
        x = res.states.smoothed
        cycle = x[:,0]
            
        freqs, X[i,:] = compute_lombscargle(t, cycle)    
        freqs2, Y[i,:] = compute_lombscargle(t, y)
        
    X = np.mean(X, axis=0)
    Y = np.mean(Y, axis=0)
    
    resp = X / Y
    return freqs, resp

def rectangular_window(omega, N):
    output = np.sin(omega*N/2) / (np.sin(omega/2)) 
    idx = np.where(np.isclose(omega, 0))
    output[idx] = N
    return output

# Define function for string formatting of scientific notation
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(np.floor(np.log10(abs(num))))
    if precision is None:
        precision = decimal_digits
    return r'$10^{{{0:d}}}$'.format(exponent)

def plot_responses(ax, omega_analytical, omega_numerical, analytical, windowed, numerical, r, dashes):
    ax.plot(omega_analytical, analytical, color='k', label='Analytical')
    ax.plot(omega_analytical, windowed, color='orange', linestyle='dashed', dashes=dashes, label='Analytical \nwindowed')
    ax.plot(omega_numerical, numerical, color='#0072B2', label='Numerical')
    ax.set_xlim(0.93, 1.07)
    ax.set_xlabel('Frequency [radians/sample]')
    ax.set_title('r = {}'.format(sci_notation(r,1)))
    return 


def main():
    t = np.arange(0, N_time)
    omega_array = np.linspace(-np.pi, np.pi, 100_000)
    window = rectangular_window(omega_array, N_time)
    omega0 = 2*np.pi / period
    
    r = r_values[0]
    resp_analytical = H_RW_amp(omega_array, omega0, r)
    resp_windowed = np.convolve(resp_analytical, window, mode='same') / len(omega_array)
    resp_windowed_scaled = np.abs(resp_windowed) / np.max(np.abs(resp_windowed))
    freqs, resp_numerical = compute_freq_response(t, r)
    
    fig, axes = plt.subplots(ncols=2, figsize=(14,6))
    ax = axes[0]
    plot_responses(ax, omega_array, freqs*2*np.pi, resp_analytical, resp_windowed_scaled, resp_numerical, r, dashes=(5,10))
    ax.set_ylabel('Magnitude response [-]')
    
    r = r_values[1]
    resp_analytical = H_RW_amp(omega_array, omega0, r)
    resp_windowed = np.convolve(resp_analytical, window, mode='same') / len(omega_array)
    resp_windowed_scaled = np.abs(resp_windowed) / np.max(np.abs(resp_windowed))
    freqs, resp_numerical = compute_freq_response(t, r)
    
    ax = axes[1]
    plot_responses(ax, omega_array, freqs*2*np.pi, resp_analytical, resp_windowed_scaled, resp_numerical, r, dashes=(5,5))
    ax.legend(loc='upper right')
    fig.savefig('../figures/Figure_4.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    return

if __name__ == '__main__':
    main()


