import numpy as np 
import matplotlib.pyplot as plt
import pdb
from astropy.time import Time
from uncertainties import unumpy as unp



#### Load in flux calibration values ####
#file = '/home/mseth2/scratch/02_23_fluxcal_results/fluxcal_results.npz'
file = '/Users/meenaseth/sidelobe-flux-calibration/fluxcal_results.npz'
widths_file = '/Users/meenaseth/sidelobe-flux-calibration/pulse_widths.npz'
#outdir = '/home/mseth2/scratch/02_23_fluxcal_results'



with np.load(file, allow_pickle=True) as data: 
    has = data['has']
    fluxes = data['fluxes']
    event_ids = data['event_ids']
    #event_times = Time(data['event_times']).mjd
    #rand_uncs = data['rand_uncs']
    #sys_uncs = data['sys_uncs']
    #total_uncs = data['total_uncs']
    #lums = data['lums']
    #lum_uncs = data['lum_uncs']
    #fluences = data['fluences']

with np.load(widths_file, allow_pickle=True) as data:
    widths = data['widths']
    #ufluxes = unp.uarray(fluxes, total_uncs)
    #fluences = ufluxes * widths

fluxes[169] = np.nan


    

## Which plots to generate? 
flux_ha = False
flux_time = False 
flux_hist = False 
fluence_hist = True 
savepdf = False

if flux_ha:
    '''
    HA vs. flux
    '''
    plt.figure(figsize =(9, 6))
    plt.errorbar(has, fluxes/10**3, yerr=total_uncs/10**3, fmt='o', markerfacecolor='blue', markeredgecolor='blue', markersize=3,
                linestyle='None', capsize=3, c='cornflowerblue', elinewidth=0.6)
    plt.ylabel("Flux Density(kJy)")
    plt.xlabel("HA (deg from meridian)")
    plt.legend()
    if savepdf:
        plt.savefig(f"{outdir}/2_HA_vs_flux_with_uncertainty.pdf")
    else: 
        plt.savefig(f"{outdir}/2HA_vs_flux_with_uncertainty.png")

if flux_time:
    '''
    Time vs. flux
    '''
    plt.figure(figsize=(18, 6))
    plt.scatter(x=event_times, y=fluxes/10**3, c=np.abs(has), cmap='viridis')
    plt.xticks([58000, 58500, 59000, 59500, 60000, 60500, 61000])
    plt.colorbar(label='HA')
    plt.ylabel("Flux (kJy)")
    plt.xlabel("Event Time (MJD)")
    if savepdf:
        plt.savefig(f"{outdir}/Flux_time.pdf")
    else:
        plt.savefig(f"{outdir}/Flux_time.png")


if flux_hist:
    '''
    Flux histogram (log scale)
        '''
    plt.figure()
    plt.hist(np.log10(fluxes), bins='auto')
    #plt.yscale('log')
    plt.ylabel("No. of pulses")
    plt.xlabel("Log(Flux (Jy))")
    if savepdf:
        plt.savefig(f"{outdir}/fluxhist_logscale.pdf")
    else:
        plt.savefig(f"{outdir}/fluxhist_logscale.png")


if fluence_hist:
    '''
    Fluence histogram (log scale)
    '''
    plt.figure()
    plt.hist(np.log10(fluences), bins='auto')
    #plt.yscale('log')
    plt.ylabel("No. of pulses")
    plt.xlabel("Log(Fluence (Jy-ms))")
    if savepdf:
        plt.savefig(f"{outdir}/fluencehist_logscale.pdf")
    else:
        plt.savefig(f"{outdir}/fluencehist_logscale.png")

