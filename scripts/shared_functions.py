import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

def H_RW(omega, r):
    den = 1 + r * (2 - 2*np.cos(omega))
    return 1 / den

def H_IRW(omega, r):
    den = 1 + r * (2 - 2*np.cos(omega))**2
    return 1 / den

def H_RW_amp(omega, omega0, r):
    num = 2*(np.cos(omega) - np.cos(omega0))**2
    den = 1 - np.cos(omega) * np.cos(omega0)
    return 1 / (1 + r * num / den)

def H_IRW_amp(omega, omega0, r):
    U_term1 = np.sin(2*omega0)**2 * (2 - 2*np.cos(4*omega))
    U_term2 = 16 * np.sin(omega0) * np.sin(2*omega0) * (np.cos(omega) - np.cos(3*omega))
    U_term3 = 16 * np.sin(omega0)**2 * (2*np.cos(2*omega) - 2)
    V = 8 * np.cos(omega) * np.cos(omega0) - 2 * np.cos(2*omega) * np.cos(2*omega0) - 6
    U = U_term1 - U_term2 - U_term3
    return 1 / (1 + r * (U / V - V))

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

def compute_windowed_freq_response(omega, H, N):
    window = rectangular_window(omega, N)
    H_windowed = np.convolve(H, window, mode='same') / len(omega)
    H_windowed = np.abs(H_windowed) / np.max(np.abs(H_windowed))
    return H_windowed

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