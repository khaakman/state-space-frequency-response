import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from statsmodels.tsa.statespace.structural import UnobservedComponents
from shared_functions import H_RW_amp, sci_notation, compute_windowed_freq_response
from matplotlib import rcParams
plt.style.use('ggplot')
rcParams.update({'font.size': 16})

### Settings ###
N_realizations = 10_000  #number of white noise realizations; lower this for less computational time
N_time = 1_000 #number of timesteps per white noise realization
period = 2*np.pi #period of periodic term, 2pi gives 1rad/sample normalized frequency
r_values = [1e3, 1e6] #noise variance ratios for subplots
###############



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


def plot_responses(ax, omega_analytical, omega_numerical, analytical, windowed, numerical, r, dashes):
    ax.plot(omega_analytical, analytical, color='k', label='Analytical')
    ax.plot(omega_analytical, windowed, color='orange', linestyle='dashed', dashes=dashes, label='Analytical \nwindowed', zorder=10)
    ax.plot(omega_numerical, numerical, color='#0072B2', label='Numerical')
    ax.set_xlim(0.93, 1.07)
    ax.set_xlabel('Frequency [radians/sample]')
    ax.set_title('NVR = {}'.format(sci_notation(r,1)))
    
    return 


def main():
    t = np.arange(0, N_time)
    omega_array = np.linspace(-np.pi, np.pi, 100_000)
    omega0 = 2*np.pi / period
    
    r = r_values[0]
    resp_analytical = H_RW_amp(omega_array, omega0, r)
    resp_windowed = compute_windowed_freq_response(omega_array, resp_analytical, N_time)
    freqs, resp_numerical = compute_freq_response(t, r)
    
    fig, axes = plt.subplots(ncols=2, figsize=(14,6))
    ax = axes[0]
    plot_responses(ax, omega_array, freqs*2*np.pi, resp_analytical, resp_windowed, resp_numerical, r, dashes=(5,10))
    ax.set_ylabel('Magnitude response [-]')
    ax.text(0.02, 0.95, 'a)', fontweight='semibold', transform=ax.transAxes)
    
    r = r_values[1]
    resp_analytical = H_RW_amp(omega_array, omega0, r)
    resp_windowed = compute_windowed_freq_response(omega_array, resp_analytical, N_time)
    freqs, resp_numerical = compute_freq_response(t, r)
    
    ax = axes[1]
    plot_responses(ax, omega_array, freqs*2*np.pi, resp_analytical, resp_windowed, resp_numerical, r, dashes=(5,5))
    ax.legend(loc='upper right')
    ax.text(0.02, 0.95, 'b)', fontweight='semibold', transform=ax.transAxes)
    fig.savefig('../figures/Figure_4.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    return

if __name__ == '__main__':
    main()


