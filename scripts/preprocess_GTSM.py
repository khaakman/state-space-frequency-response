import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


### OPTIONS ###
stations = ['Vlissingen', 'Hoek van Holland', 'IJmuiden', 'Den Helder', 'Harlingen', 'Delfzijl']
years = np.arange(1890, 2022) #for which period you want to make correction
###############

years_GTSM = np.arange(1950, 2023) #GTSM surge is from Jan 1950 until Dec 2022
fn = '../data/GTSM/gtsm_surge_annual_mean_main_stations_2023.csv'
data = pd.read_csv(fn) 

for station in stations:
    data_station = data[data.name == station]
    surge = data_station.surge.to_numpy() * 100 #m to cm
    extended_surge = np.zeros(len(years)) # empty array to store surge for chosen years
    mean_surge = np.mean(surge) #mean surge used to correct years before 1950
    
    for i in range(len(extended_surge)):
        if years[i] not in years_GTSM:
            extended_surge[i] = mean_surge
        else:
            GTSM_idx = np.where(years_GTSM == years[i])[0][0]
            extended_surge[i] = surge[GTSM_idx] #np.mean(surge[12*GTSM_idx:12*(GTSM_idx+1)])
        
    if station == 'Hoek van Holland':
        savename = 'HvH'
    elif station == 'Den Helder':
        savename = 'DenHelder'
    else:
        savename = station
        
    np.save('../processed_data/GTSM_surge_{}_yearly.npy'.format(savename), extended_surge)
    
    plt.figure()
    plt.plot(years, extended_surge)
    plt.ylabel('Height [cm]')