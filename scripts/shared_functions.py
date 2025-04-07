#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 17:08:57 2025

@author: kbhaakman
"""

import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

def H_RW(omega, r):
    den = 1 + r * (2 - 2*np.cos(omega))
    return 1 / den

def H_IRW(omega, r):
    den = 1 + r * (2 - 2*np.cos(omega))**2
    return 1 / den

def H_RW_amp(omega, omega0, r):
    num = 2 - np.cos(omega0) * 2 * np.cos(omega)
    den = 2 + 4*r + 2*r*np.cos(2*omega0) - (4*r + 1) * np.cos(omega0)*2*np.cos(omega) + r*(2*np.cos(2*omega))
    return num / den

def H_IRW_amp(omega, omega0, r):
    num = 1
    term1 = np.cos(2*omega0) * 2*np.cos(2*omega)
    term2 = 4 * np.cos(omega0) * 2*np.cos(omega)
    term3 = np.sin(2*omega0) * -2j * np.sin(2*omega) #(z**(-2) - z**2)
    term4 = 4 * np.sin(omega0) * -2j * np.sin(omega) #(z**(-1) - z)
    term5 = np.sin(2*omega0) * 2j * np.sin(2*omega) #(z**2 - z**(-2))
    
    den = 1 + r * (6 + term1 - term2 + (term3 - term4) * (term5 + term4) / (term2 - term1 - 6))
    return num / den

def H_IRW_trend_AR1_errors(omega, r_eta, phi):
    b = phi / (phi**2 - 1)
    c = (phi**2 + 1) / (1 - phi**2)
    num = c + b*2*np.cos(omega)
    den = c + 6*r_eta + r_eta*2*np.cos(2*omega) + (b - 4*r_eta) * 2 * np.cos(omega)
    return num / den

def H_M(omega, omega0, r_eta, r_xi):
    D =  2 * np.cos(omega0)*np.cos(omega) - 2 +  (np.sin(omega0)**2 * (2 - 2*np.cos(2*omega))) / (2 - 2 * np.cos(omega0) * np.cos(omega))
    nabla4 = -(2 - 2*np.cos(omega))**2
    num = 1
    den = 1 - r_eta * nabla4 + r_eta * nabla4 / (r_xi * D)
    return num/den

def H_C(omega, omega0, r_eta, r_xi):
    D =  2 * np.cos(omega0)*np.cos(omega) - 2 +  (np.sin(omega0)**2 * (2 - 2*np.cos(2*omega))) / (2 - 2 * np.cos(omega0) * np.cos(omega))
    nabla4 = -(2 - 2*np.cos(omega))**2
    num = 1
    den = 1 + r_xi * D / (r_eta * nabla4) - r_xi * D
    return num/den

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

def rectangular_window(omega, N):
    output = np.sin(omega*N/2) / (np.sin(omega/2)) 
    idx = np.where(np.isclose(omega, 0))
    output[idx] = N
    return output

def load_PSMSL_data(fn, t0, t1):
    data = np.loadtxt(fn, delimiter=';', usecols=[0,1])
    t = data[:,0]
    y = data[:,1] / 10 #convert mm to cm
    y -= np.nanmean(y)
    
    # Only keep data between t0 and t1
    idx = np.where(t >= t0)
    y = y[idx]
    t = t[idx]
    
    idx = np.where(t < t1)
    y = y[idx]
    t = t[idx]
    return t, y

def correct_surge(y, station):
    surge_fn = '../processed_data/GTSM_surge_{}_yearly.npy'.format(station)
    surge = np.load(surge_fn)
    y -= surge
    return y

def define_model(t, y):
    #Define nodal regressors
    omega_nodal = 2*np.pi / 18.613
    X = np.zeros((len(t), 2))
    nodal_cos = np.cos(omega_nodal * t)
    nodal_sin = np.sin(omega_nodal * t)
    X[:,0] = nodal_cos
    X[:,1] = nodal_sin
 
    model = UnobservedComponents(y, level=True, trend=True, stochastic_level=False,
                                 stochastic_trend=True,
                                 autoregressive=1, exog=X, mle_regression=True, irregular=False)
    
    return model