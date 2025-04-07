#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 13:33:29 2025

@author: kbhaakman
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from shared_functions import H_IRW, rectangular_window

### Settings ###
Nomega = 50_000
r_vec = np.logspace(1, 13, 50)
################


def compute_windowed_freq_resp(freq_resp, window):
    windowed_freq_resp = np.convolve(freq_resp, window, mode='full') / Nomega
    windowed_freq_resp = np.abs(windowed_freq_resp)[Nomega//2 : Nomega+Nomega//2] / np.max(np.abs(windowed_freq_resp))
    windowed_freq_resp = windowed_freq_resp[Nomega // 2 :]
    return windowed_freq_resp

def compute_cutoff_freqs(Ntime, r_vec):
    # Define frequencies
    omega_array = np.arange(-np.pi, np.pi, 2*np.pi/Nomega)[:Nomega] #neg and pos frequencies
    trunc_omega_array = omega_array[Nomega//2 : Nomega+Nomega//2] #positive frequencies
    
    # Set empty arrays to store cut-off frequencies
    cutoff_freqs = np.zeros(len(r_vec))
    windowed_cutoff_freqs = np.zeros(len(r_vec))
    
    # Compute DTFT of rectangular window
    window = rectangular_window(omega_array, Ntime)
    
    for i in range(len(r_vec)): #for each NVR, compute cut-off frequency
        # For analytical frequency response
        freq_resp = np.abs(H_IRW(omega_array, r_vec[i]))
        trunc_freq_resp = freq_resp[Nomega // 2:]
        idx = np.where(trunc_freq_resp < 1/np.sqrt(2))[0][0]
        cutoff_freqs[i] = trunc_omega_array[idx]
        
        # For windowed frequency response
        windowed_freq_resp = compute_windowed_freq_resp(freq_resp, window)
        windowed_idx = np.where(windowed_freq_resp < 1/np.sqrt(2))[0][0]
        windowed_cutoff_freqs[i] = trunc_omega_array[windowed_idx]
    
    # Convert radial to linear frequency
    cutoff_freqs = cutoff_freqs / (2*np.pi)
    windowed_cutoff_freqs = windowed_cutoff_freqs / (2*np.pi)
    return cutoff_freqs, windowed_cutoff_freqs



def make_figure():
    plt.style.use('ggplot')
    rcParams.update({'font.size': 18})
    
    # 50 years
    Ntime = 50 #50 timesteps (50 years with yearly measurements)
    cutoff_freqs, windowed_cutoff_freqs = compute_cutoff_freqs(Ntime, r_vec)
    
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    ax = axes[0]
    ax.plot(r_vec, windowed_cutoff_freqs, label='Windowed frequency response')
    ax.plot(r_vec, cutoff_freqs, label='Analytical frequency response')
    ax.set_title('50 years')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Cut-off frequency [cpy]')
    ax.set_xlabel(r'NVR [-]')
    ax.axhline(1/Ntime, color='k', label='Frequency resolution')
    ax.text(5, 1.1e-4, 'a)', fontweight='semibold')
    #ax.legend(loc='upper center')


    # 132 years
    Ntime = 132 #50 timesteps (132 years with yearly measurements)
    cutoff_freqs, windowed_cutoff_freqs = compute_cutoff_freqs(Ntime, r_vec)
    
    ax = axes[1]
    ax.plot(r_vec, windowed_cutoff_freqs, label='Windowed frequency response')
    ax.plot(r_vec, cutoff_freqs, label='Analytical frequency response')
    ax.set_title('132 years')
    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.set_ylabel('Cutoff frequency [cpy]')
    ax.set_xlabel(r'NVR [-]')
    ax.axhline(1/Ntime, color='k', label='Frequency resolution')
    ax.legend(loc='upper center', fontsize=14)
    ax.text(5, 1.1e-4, 'b)', fontweight='semibold')
    plt.savefig('../figures/Figure_5.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    return

def main():
    make_figure()
    
    return


if __name__ == '__main__':
    main()

