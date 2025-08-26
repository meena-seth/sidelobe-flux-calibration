import numpy as np 
import matplotlib.pyplot as plt
import pdb

#### Load in flux calibration values ####
file = np.load('/arc/projects/chime_frb/mseth/rfi_corrected_calibration_1.npz', allow_pickle=True)

mjd_all = file['mjds']
ha_all = file['has']

ha_all[ha_all > -20] = np.nan

scaled_flux_all = file['scaled_fluxes']
fluences_all = file['fluence']
luminosities_all = file['peak_luminosity']
flux_peakfreq = file['peakfreq_fluxes']

#file_2 = np.load('/')
#mjd_2 = file_2['mjd']
#ha_2 = file_2['ha']
#scaled_flux_all_2= file_2['scaled_flux']
#fluences_all_2 = file_2['fluence']
#luminosities_all_2 = file_2['peak_luminosity']

#### Pick out "good" observations ####


#good_obs = np.array([0, 1, 2, 5, 8, 12, 13, 15, 17, 19, 21, 23, 26, 28, 30, 31, 33, 35, 46, 47, 49, 59, 60])

#good_obs = np.array([0, 1, 19, 2, 23, 26, 59, 60, 8])
f"""
good_obs = np.array([0, 1, 2, 3, 4, 5, 6, 8, 
                     12, 13, 14, 15, 17, 19, 
                     20, 21, 23, 26, 28, 
                     31, 33, 35, 38,  
                     42, 43, 46, 47, 48, 49, 
                     50, 51, 53, 57, 59, 60])

scaled_flux = scaled_flux_all[good_obs]
fluence = fluences_all[good_obs]
peak_luminosity = luminosities_all[good_obs]
ha = ha_all[good_obs]
mjd = mjd_all[good_obs]
"""

#######################################

# Flux
plt.figure()
plt.hist(scaled_flux_all/10**3, bins='auto')
plt.ylabel("No. of pulses")
plt.xlabel("Flux (kJy)")
#plt.savefig("/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/summary_plots/flux_scaled")

plt.figure()
plt.hist(scaled_flux_all/10**3, bins='auto')
plt.yscale('log')
plt.xscale('log')
plt.ylabel("No. of pulses")
plt.xlabel("Flux (kJy)")
#plt.savefig("/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/summary_plots/loglog_flux_scaled")

# Flux from max frequency
plt.figure()
plt.hist(flux_peakfreq/10**3, bins='auto')
plt.ylabel("No. of pulses")
plt.xlabel("Flux (kJy)")
#plt.savefig("/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/summary_plots/maxfreq_flux")


# Fluence
plt.figure()
plt.hist(fluences_all*5/10**3, bins=5)
plt.ylabel("No. of pulses")
plt.xlabel("Fluence (Jy-s)")
#plt.savefig("/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/summary_plots/Fluence_scaled")

# Luminosity
plt.figure()
plt.hist(luminosities_all, bins='auto')
plt.ylabel("No. of pulses")
plt.xlabel("Peak luminosity (erg/s/Hz)")
#plt.savefig("/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/summary_plots/Peak_luminosities")


# HA vs. flux 
plt.figure()
plt.scatter(ha_all, scaled_flux_all/10**3)
plt.ylabel("Flux Density(kJy)")
plt.xlabel("HA (deg from zenith)")
#plt.savefig("/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/summary_plots/HA_vs_flux")

#Flux vs. averaging over HAs
HAs = np.arange(2, 19)
fluxes = np.array([52096.68325609861, 52270.06567006741, 51792.12609320032, 51595.514901003204, 51832.69808801434, 51585.64186349405, 52115.78416289081, 51762.51520175683, 51883.451171568246, 51672.30440841518, 51842.395816000804, 51556.188084447815, 51928.3719295107, 51457.57744949451, 51859.135244271296, 51721.598602621816, 51918.34978317152])

plt.figure()
plt.scatter(HAs, fluxes/10**3)
plt.scatter(x=1, y=53418.59567092105/1000, color='r', label="HA of observation=-79.6")
plt.scatter(x=12, y=51842.395816000804/1000, color='b', label="Averaged over 1 degree")
plt.ylabel("Flux Density(kJy)")
plt.xlabel("no. of HAs averaged over")
plt.legend()
plt.savefig("/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/summary_plots/averaging_HAs_vs_flux")