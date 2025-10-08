import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from shared_functions import load_PSMSL_data, correct_surge, define_model
plt.style.use('ggplot')
rcParams.update({'font.size': 16})

### OPTIONS ###
t0 = 1890 #inclusive
t1 = 2022 #exclusive
correct_surge_GTSM = True
r_vec = [1e2, 1e4]
################

stations = ['Delfzijl']
fn_base = '../data/PSMSL/{}_yearly.rlrdata'
filenames = [fn_base.format(station) for station in stations]

def compute_single_station(fn, NVR):
    station = fn.split('/')[-1].split('_')[0]
    t, y = load_PSMSL_data(fn, t0, t1)
    if correct_surge_GTSM:
        y = correct_surge(y, station)
    
        
    model = define_model(t, y)
    res = model.fit(disp=False)
    
    phi = res.params[2] #AR(1) param
    var_AR1 = res.params[1] #AR(1) noise variance
    beta_x1 = res.params[3] #nodal coefficients
    beta_x2 = res.params[4]
    
    constraints = {}
    constraints['sigma2.trend'] = (1 - phi) * var_AR1 / NVR
    constraints['sigma2.ar'] = var_AR1
    constraints['ar.L1'] = phi
    constraints['beta.x1'] = beta_x1
    constraints['beta.x2'] = beta_x2
    res = model.fit_constrained(constraints)

    print(res.summary())
    
    trend = res.states.smoothed[:,0]
    trend_sigma = np.sqrt(res.smoothed_state_cov[0,0,:])
    
    return trend, trend_sigma, y


SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

def to_superscript_power(num):
    if num == 0:
        return "0"
    exponent = np.log10(abs(num))
    if exponent.is_integer():
        sign = "-" if num < 0 else ""
        exp_str = str(int(exponent)).translate(SUPERSCRIPTS)
        return f"{sign}10{exp_str}"
    else:
        # fallback to normal scientific notation for non-exact powers
        exponent = int(np.floor(np.log10(abs(num))))
        base = num / (10 ** exponent)
        exp_str = str(exponent).translate(SUPERSCRIPTS)
        return f"{base:.2f}×10{exp_str}"
    
def plot(trends, trend_sigmas):
    
    plt.figure(figsize=(14,6))
    colors = []
    for i, NVR in enumerate(r_vec):
        trend = trends[i,:]
        trend -= np.mean(trend)
        sigmas = trend_sigmas[i,:]
        line = plt.plot(years, trend, label='NVR = {}'.format(to_superscript_power(NVR)))
        colors.append(line[0].get_color())
        lower = trend - 1.96 * sigmas
        upper = trend + 1.96 * sigmas
        plt.fill_between(years, lower, upper, alpha=0.15)
    plt.legend(reverse=True, ncol=5, fontsize=14)
    plt.xlabel('Time [years]')
    plt.ylabel('Sea level [cm]')
    plt.xlim(1980, 2022)
    plt.ylim(bottom=0, top=24)
    plt.savefig('../figures/peer_review_figure.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    return


def main():
    global years
    years = np.arange(t0, t1)
    
    observations = np.zeros((len(r_vec), len(years)))
    trends = np.zeros((len(r_vec), len(years)))
    trend_sigmas = np.zeros_like(trends)
    
    for i, NVR in enumerate(r_vec):
        fn = filenames[0]
        trends[i,:], trend_sigmas[i,:], observations[i,:] = compute_single_station(fn, NVR)
    
    plot(trends, trend_sigmas)
    
    return

if __name__ == '__main__':
    main()