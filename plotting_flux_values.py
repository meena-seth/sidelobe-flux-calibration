import numpy as np 
import matplotlib.pyplot as plt
import pdb

#### Load in flux calibration values ####
file = np.load('/arc/projects/chime_frb/mseth/nrao/flux_calibration/rfi_corrected_calibration.npz', allow_pickle=True)

mjd_all = file['mjds']
ha_all = file['has']
scaled_flux_all = file['scaled_fluxes']
fluences_all = file['fluence']
luminosities_all = file['peak_luminosity']

file_maxfreq = np.load('/arc/projects/chime_frb/mseth/nrao/flux_calibration/rfi_corrected_calibration_maxfreqs.npz')
flux_maxfreq = file_maxfreq['scaled_fluxes']

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

pdb.set_trace()
#######################################

# Flux
plt.figure()
plt.hist(scaled_flux_all/10**3, bins='auto')
plt.ylabel("No. of pulses")
plt.xlabel("Flux (kJy)")
plt.savefig("a_flux_scaled")

plt.figure()
plt.hist(scaled_flux_all/10**3, bins='auto')
plt.yscale('log')
plt.xscale('log')
plt.ylabel("No. of pulses")
plt.xlabel("Flux (kJy)")
plt.savefig("a_loglog_flux_scaled")

plt.figure()
plt.hist(flux_maxfreq/10**3, bins='auto')
plt.ylabel("No. of pulses")
plt.xlabel("Flux (kJy)")
plt.savefig("a_maxfreq_flux")


# Fluence
plt.figure()
plt.hist(fluences_all*5/10**3, bins=5)
plt.ylabel("No. of pulses")
plt.xlabel("Fluence (Jy-s)")
#plt.savefig("a_New_Fluence_scaled")

# Luminosity
plt.figure()
plt.hist(luminosities_all, bins='auto')
plt.ylabel("No. of pulses")
plt.xlabel("Peak luminosity (erg/s/Hz)")
#plt.savefig("a_New_Peak_luminosities")



# HA vs. flux 
plt.figure()
plt.scatter(ha_all, scaled_flux_all/10**3)
plt.ylabel("Flux Density(kJy)")
plt.xlabel("HA (deg from zenith)")
#plt.savefig("a_New_HA_vs_brightness")

pdb.set_trace()
#Flux vs. averaging over HAs
HAs = np.arange(1, 13)
fluxes = np.array([31610.744352056296, 31227.94811519268, 31180.75070503412, 31335.05908287224, 31599.972619751497, 31458.59920473724, 31674.614225323014, 31361.83141869902, 31491.369605496762, 31194.537276722192, 31320.065288796774, 31012.57054185125])

pdb.set_trace()
plt.figure()
plt.scatter(HAs, fluxes/10**3)
plt.ylabel("Flux Density(kJy)")
plt.xlabel("no. of HAs averaged over")
#plt.savefig("averaging_HAs_vs_flux")