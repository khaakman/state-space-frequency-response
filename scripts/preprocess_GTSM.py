#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 16:51:31 2025

@author: kbhaakman
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


### OPTIONS ###
station = 'Vlissingen'
years = np.arange(1890, 2022) #for which period you want to make correction
###############


years_GTSM = np.arange(1950, 2023) #GTSM surge is from Jan 1950 until Dec 2022
fn = '../data/GTSM/gtsm_surge_monthly_mean_main_stations_2023.csv'
data = pd.read_csv(fn) 
data = data[data.name == station]
surge = data.surge.to_numpy() * 100 #m to cm
surge_yearly = np.zeros(len(years))
mean_surge = np.mean(surge) #mean surge used to correct years before 1950

for i in range(len(surge_yearly)):
    if years[i] not in years_GTSM:
        surge_yearly[i] = mean_surge
    else:
        GTSM_idx = np.where(years_GTSM == years[i])[0][0]
        surge_yearly[i] = np.mean(surge[12*GTSM_idx:12*(GTSM_idx+1)])
    
if station == 'Hoek van Holland':
    savename = 'HvH'
elif station == 'Den Helder':
    savename = 'DenHelder'
else:
    savename = station
    
np.save('../processed_data/GTSM_surge_{}_yearly.npy'.format(savename), surge_yearly)

plt.figure()
plt.plot(years, surge_yearly)
plt.ylabel('Height [cm]')

