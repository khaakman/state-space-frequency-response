#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 13:18:27 2025

@author: kbhaakman
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from shared_functions import H_RW, H_IRW, H_RW_amp, H_IRW_amp, H_C, H_M, H_IRW_trend_AR1_errors, sci_notation
plt.style.use('ggplot')
rcParams.update({'font.size': 16})




omega_array = np.arange(0, np.pi, np.pi/10_000)
linestyles = ['solid', 'dashed', 'dashdot']
r_vec = [1e5, 1e3, 1e1]
r_vec_IRW = r_vec


# Figure 1
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
for i, r in enumerate(r_vec):
    mag_resp_RW = np.abs(H_RW(omega_array, r))
    axes[0].plot(omega_array, mag_resp_RW, label=r'r = $\sigma^2_{{\varepsilon}} / \sigma^2_{{\eta}}$ = {}'.format(sci_notation(r, 1)), linestyle=linestyles[i])
    
    r_IRW = r_vec_IRW[i]
    mag_resp_IRW = np.abs(H_IRW(omega_array, r_IRW))
    axes[1].plot(omega_array, mag_resp_IRW, label=r'r = $\sigma^2_{{\varepsilon}} / \sigma^2_{{\eta}}$ = {}'.format(sci_notation(r_IRW, 1)), linestyle=linestyles[i])
axes[0].set_xlim(0, 0.4)
axes[1].set_xlim(0, 0.4)
axes[0].set_xlabel('Frequency [rad/sample]')
axes[1].set_xlabel('Frequency [rad/sample]')
axes[0].set_ylabel('Magnitude response [-]')
axes[0].legend(loc='upper right')
#axes[1].legend(loc='lower left')
axes[0].set_title(r'$\mathbf{(a)}$ Random walk')
axes[1].set_title(r'$\mathbf{(b)}$ Integrated random walk')
fig.savefig('../figures/Figure_1.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)


# Figure 2
omega0 = 1
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
for i, r in enumerate(r_vec):
    mag_resp_RW = np.abs(H_RW_amp(omega_array, omega0, r))
    #axes[0].plot(omega_array, mag_resp_RW, label=r'r = $\sigma^2_{{\varepsilon}} / \sigma^2_{{\eta}}$ = {}'.format(sci_notation(r, 1)), linestyle=linestyles[i])
    axes[0].plot(omega_array, mag_resp_RW, label=r'r = {}'.format(sci_notation(r, 1)), linestyle=linestyles[i])
    
    r_IRW = r_vec_IRW[i]
    mag_resp_IRW = np.abs(H_IRW_amp(omega_array, omega0, r_IRW))
    #axes[1].plot(omega_array, mag_resp_IRW, label=r'r = $\sigma^2_{{\varepsilon}} / \sigma^2_{{\eta}}$ = {}'.format(sci_notation(r_IRW, 1)), linestyle=linestyles[i])
    axes[1].plot(omega_array, mag_resp_IRW, label=r'r = {}'.format(sci_notation(r_IRW, 1)), linestyle=linestyles[i])
axes[0].set_xlim(0.5, 1.5)
axes[1].set_xlim(0.5, 1.5)
axes[0].set_xlabel('Frequency [rad/sample]')
axes[1].set_xlabel('Frequency [rad/sample]')
axes[0].set_ylabel('Magnitude response [-]')
axes[0].legend(loc='upper right')
#axes[1].legend(loc='upper right')
axes[0].set_title(r'$\mathbf{(a)}$ RW amplitude')
axes[1].set_title(r'$\mathbf{(b)}$ IRW amplitude')
fig.savefig('../figures/Figure_2.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)


# Figure 3
omega0 = 2*np.pi/75
r_eta = 1e4
r_xi = 1e3

mag_resp_C = np.abs(H_C(omega_array, omega0, r_eta, r_xi))
mag_resp_M = np.abs(H_M(omega_array, omega0, r_eta, r_xi))
mag_resp_IRW = np.abs(H_IRW(omega_array, r_eta))
mag_resp_RW_amp = np.abs(H_RW_amp(omega_array, omega0, r_xi))

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 6))
ax = axes[0]
ax.plot(omega_array, mag_resp_C, label=r'r$_\xi$ = {}'.format(sci_notation(r_xi, 1)), color='b')
ax.plot(omega_array, mag_resp_M, label=r'r$_\eta$ = {}'.format(sci_notation(r_eta, 1)), color='orange')
ax.plot(omega_array, mag_resp_IRW, linestyle='dashed', color='orange')
ax.plot(omega_array, mag_resp_RW_amp, linestyle='dashed', color='blue')
ax.set_xlim(0, 0.3)
#ax.set_xlabel('Frequency [rad/sample]')
ax.set_ylabel('Magnitude response [-]')
ax.legend(loc='upper right')
#axes[1].legend(loc='upper right')
ax.set_title('IRW trend + RW amp cycle')

omega0 = 2*np.pi/13
mag_resp_C = np.abs(H_C(omega_array, omega0, r_eta, r_xi))
mag_resp_M = np.abs(H_M(omega_array, omega0, r_eta, r_xi))
mag_resp_IRW = np.abs(H_IRW(omega_array, r_eta))
mag_resp_RW_amp = np.abs(H_RW_amp(omega_array, omega0, r_xi))

ax = axes[1]
ax.plot(omega_array, mag_resp_C, label=r'r$_\xi$ = {}'.format(sci_notation(r_xi, 1)), color='b')
ax.plot(omega_array, mag_resp_M, label=r'r$_\eta$ = {}'.format(sci_notation(r_eta, 1)), color='orange')
ax.plot(omega_array, mag_resp_IRW, linestyle='dashed', color='orange')
ax.plot(omega_array, mag_resp_RW_amp, linestyle='dashed', color='blue')
ax.set_xlim(0, 0.6)
ax.set_xlabel('Frequency [rad/sample]')
ax.set_ylabel('Magnitude response [-]')
ax.legend(loc='upper right')
#axes[1].legend(loc='upper right')
#ax.set_title('IRW trend + RW amp cycle')
fig.savefig('../figures/Figure_3.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)



# Figure A1
r_eta = 1e4
phi_list = [-0.8, -0.3, 0.3, 0.8]

mag_resp_IRW = np.abs(H_IRW(omega_array, r_eta))

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(14, 6))


axes.plot(omega_array, mag_resp_IRW, linestyle='solid', color='orange', label=r'$\phi$ = 0')
for i, phi in enumerate(phi_list):
    mag_resp_IRW_AR1 =  np.abs(H_IRW_trend_AR1_errors(omega_array, r_eta, phi))
    axes.plot(omega_array, mag_resp_IRW_AR1, linestyle='dashed', label=r'$\phi$ = {}'.format(phi))
axes.set_xlim(0, 0.3)
axes.set_xlabel('Frequency [rad/sample]')
axes.set_ylabel('Magnitude response [-]')
axes.legend(loc='upper right')
#axes[1].legend(loc='upper right')
axes.set_title('IRW trend + AR(1) errors')
fig.savefig('../figures/Figure_A1.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)