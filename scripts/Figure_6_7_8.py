import numpy as np
import matplotlib.pyplot as plt
from shared_functions import H_IRW_trend_AR1_errors, load_PSMSL_data, correct_surge, define_model, compute_windowed_freq_response
from matplotlib import rcParams
plt.style.use('ggplot')
rcParams.update({'font.size': 16})

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


def compute_single_station(fn):
    station = fn.split('/')[-1].split('_')[0]
    t, y = load_PSMSL_data(fn, t0, t1)
    if correct_surge_GTSM:
        y = correct_surge(y, station)
    
        
    model = define_model(t, y)
    res = model.fit(disp=False)
    #res.plot_diagnostics()
    print(res.summary())
    
    trend = res.states.smoothed[:,0]
    trend_sigma = np.sqrt(res.smoothed_state_cov[0,0,:])
    
    return trend, trend_sigma

def compute_loglikelihood_function(fn, r_vec):
    station = fn.split('/')[-1].split('_')[0]
    t, y = load_PSMSL_data(fn, t0, t1)
    if correct_surge_GTSM:
        y = correct_surge(y, station)
    
    model = define_model(t, y)
    
    ll_vec = np.zeros(len(r_vec))
    
    res = model.fit(disp=False)
    phi_hat = res.params[2]
    sigma2_eps = (1 - phi_hat**2) * res.params[1]
    r_hat = sigma2_eps / res.params[0]
    
    constraints = {}
    constraints['sigma2.ar'] = res.params[1]
    constraints['ar.L1'] = res.params[2]
    constraints['beta.x1'] = res.params[3]
    constraints['beta.x2'] = res.params[4]
    
    for i, r_eta in enumerate(r_vec):
        constraints['sigma2.trend'] = res.params[1] / r_eta
        res2 = model.fit_constrained(constraints, disp=False)
        ll_vec[i] = res2.llf
            
    return ll_vec, r_hat, phi_hat


def plot_Figure_6(trends, trend_sigmas):
    offset = 12
    plt.figure(figsize=(14,6))
    colors = []
    for i, station in enumerate(stations):
        trend = trends[i,:]
        trend -= np.mean(trend)
        sigmas = trend_sigmas[i,:]
        line = plt.plot(years, trend + offset*i, label=title_names[i])
        colors.append(line[0].get_color())
        lower = trend + offset*i - 1.96 * sigmas
        upper = trend + offset*i + 1.96 * sigmas
        plt.fill_between(years, lower, upper, alpha=0.3)
    plt.legend(reverse=True, ncol=5, fontsize=14)
    plt.xlabel('Time [years]')
    plt.ylabel('Sea level + arbitrary offset [cm]')
    plt.ylim(bottom=-20, top=90)
    plt.savefig('../figures/Figure_6.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    return colors


def plot_Figure_7(ll_vecs, r_hats):
    plt.figure(figsize=(8,6))
    for i, station in enumerate(stations):
        if station in ['DenHelder', 'IJmuiden', 'Vlissingen', 'HvH', 'Harlingen', 'Delfzijl']:
            ll = ll_vecs[i,:]

            plt.plot(r_vec, ll, label=title_names[i])
            
            plt.scatter(r_hats[i], np.max(ll), s=50)

    plt.legend(loc='lower right')
    plt.xlabel('NVR [-]')
    plt.xscale('log')
    plt.xlim(right=1e13)
    plt.ylabel('Loglikelihood [-]')
    plt.savefig('../figures/Figure_7.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    return


def plot_Figure_8(stations, r_hats, phi_hats, colors):
    omega = np.linspace(-np.pi, np.pi, 100_000)
    
    fig, ax = plt.subplots(figsize=(14,6))
    for i, station in enumerate(stations):
        idx = np.where(np.array(title_names) == station)[0][0]
        r_hat = r_hats[idx]
        phi_hat = phi_hats[idx]
        H = H_IRW_trend_AR1_errors(omega, r_hat, phi_hat)
        H_windowed = compute_windowed_freq_response(omega, H, len(years))
        
        ax.plot(omega/(2*np.pi), H_windowed, label=station, linestyle='solid', linewidth=2, color=colors[idx])
    ax.set_xlabel('Frequency [cpy]')

    ax.set_xlim(0, 0.06)
    ax.set_ylabel('Magnitude response [-]')
    plt.axvline(1/18.613, linewidth=2, linestyle='dashed', label='Nodal cycle')
    plt.axvspan(1/70, 1/50, alpha=0.2, color='r', label='AMO')
    ax.legend()
    plt.savefig('../figures/Figure_8.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    return
    

def main():
    global years
    years = np.arange(t0, t1)
    
    ### Figure 6
    trends = np.zeros((len(filenames), len(years)))
    trend_sigmas = np.zeros_like(trends)
    
    for i, fn in enumerate(filenames):
        trends[i,:], trend_sigmas[i,:] = compute_single_station(fn)
    
    colors = plot_Figure_6(trends, trend_sigmas)
    
    
    ### Figure 7
    ll_vecs = np.zeros((len(filenames), len(r_vec)))
    r_hats = np.zeros(len(filenames))
    phi_hats = np.zeros(len(filenames))
    for i, fn in enumerate(filenames):
        ll_vecs[i,:], r_hats[i], phi_hats[i] = compute_loglikelihood_function(fn, r_vec)
        
    plot_Figure_7(ll_vecs, r_hats)
    
    
    ### Figure 8
    stations = ['IJmuiden', 'Harlingen', 'Den Helder']
    
    plot_Figure_8(stations, r_hats, phi_hats, colors)
    
    return


if __name__ == '__main__':
    main()