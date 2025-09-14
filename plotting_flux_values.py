import numpy as np 
import matplotlib.pyplot as plt
import pdb

#### Load in flux calibration values ####
file = np.load('/arc/projects/chime_frb/mseth/error_rfi_corrected_calibration.npz', allow_pickle=True)

mjd_all = file['mjds']
ha_all = file['has']

ha_all[ha_all > -20] = np.nan

scaled_flux_all = file['scaled_fluxes']
fluences_all = file['fluence']
luminosities_all = file['peak_luminosity']
#flux_peakfreq = file['peakfreq_fluxes']

sys_errors = np.array([file['lowersys_errors'], file['uppersys_errors']])
random_errors=file['random_errors']
sys_errors[sys_errors==0]=np.max(sys_errors)
combined_error = np.sqrt(np.square(sys_errors) + np.square(random_errors))

sys_lumerrors = np.array([file['lowersys_lumerror'], file['uppersys_lumerror']])
sys_lumerrors[sys_lumerrors==0]=np.max(sys_lumerrors)
ran_lumerrors = np.array([file['lowerran_lumerror'], file['upperran_lumerror']])
combined_lumerror = np.sqrt(np.square(sys_lumerrors) + np.square(ran_lumerrors))


sys_fluerrors = np.array([fluences_all - file['lowersys_fluences'], file['uppersys_fluences'] - fluences_all])
sys_fluerrors[sys_fluerrors==0]=np.max(sys_fluerrors)
ran_fluerrors = np.array([fluences_all - file['lowerran_fluences'], file['upperran_fluences'] - fluences_all])
combined_fluerror = np.sqrt(np.square(sys_fluerrors) + np.square(ran_fluerrors))


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

# Flux histogram 
flux_hist, bin_edges = np.histogram(scaled_flux_all/10**3, bins=14)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

plt.figure()
plt.bar(bin_centers, flux_hist, width=bin_width, align='center')
#plt.scatter(bin_centers, flux_hist)
#plt.yscale('log')
#plt.xscale('log')
plt.ylabel('Number of Pulses')
plt.xlabel('Flux Density (kJy)')
#plt.savefig("/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/tests/loglogscatterflux2")


# Fluence histogram 
fluence_hist, bin_edges = np.histogram(fluences_all/10**3)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

plt.figure()
#plt.bar(bin_centers, fluence_hist, width=bin_width, align='center')
plt.hist(fluences_all, bins='auto')
plt.ylabel('Number of Pulses')
plt.xlabel('Fluence (Jy-s)')
#plt.savefig("/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/summary_plots/Fluence_hist")

# Luminosity histogram 
lum_hist, bin_edges = np.histogram(fluences_all/10**3)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

plt.figure()
#plt.bar(bin_centers, lum_hist, width=bin_width, align='center')
plt.hist(luminosities_all, bins='auto')
plt.ylabel('Number of Pulses')
plt.xlabel('Luminosity (ergs/s/Hz)')
#plt.savefig("/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/summary_plots/Luminosity_hist")




## HA vs Flux (with errors) ##

#Combined error
plt.figure()
plt.ylabel("Flux Density (kJy)")
plt.xlabel("HA (deg away from meridian)")
plt.scatter(ha_all, scaled_flux_all/10**3, s=15)
plt.errorbar(ha_all, scaled_flux_all/10**3, yerr=combined_error/1000, capsize=2, ls='none')
#plt.savefig("/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/summary_plots/combined_error_HA_vs_flux")


#Systematic error only
plt.figure()
plt.scatter(ha_all, scaled_flux_all/10**3, s=15)
plt.errorbar(ha_all, scaled_flux_all/10**3, yerr=sys_errors/1000, capsize=2, ls='none', label='Random error')
#plt.savefig("/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/summary_plots/sys_error_HA_vs_flux")

#Random error only
plt.figure()
plt.scatter(ha_all, scaled_flux_all/10**3, s=15)
plt.errorbar(ha_all, scaled_flux_all/10**3, yerr=random_errors/1000, capsize=2, ls='none', label='Random error')
#plt.savefig("/arc/projects/chime_frb/mseth/plots/averaged_holography_calibration/summary_plots/ran_error_HA_vs_flux")


#Flux vs. averaging over HAs
HAs = np.arange(1, 18)         # 1..17
deg_per = 0.0972
tick_positions = HAs[::2]      # e.g. [1,5,9,13,17]
tick_labels = [f"{deg_per * i:.2f}" for i in tick_positions]

fluxes = np.array([52096.68325609861, 52270.06567006741, 51792.12609320032, 51595.514901003204, 51832.69808801434, 51585.64186349405, 52115.78416289081, 51762.51520175683, 51883.451171568246, 51672.30440841518, 51842.395816000804, 51556.188084447815, 51928.3719295107, 51457.57744949451, 51859.135244271296, 51721.598602621816, 51918.34978317152])


plt.figure()
plt.scatter(HAs, fluxes/10**3, color='gray')
plt.scatter(x=1, y=53418.59567092105/1000, color='r', label="HA of observation=-79.6")
plt.xticks(tick_positions, tick_labels)

#plt.scatter(x=12, #y=51842.395816000804/1000, color='b', label="Averaged over 1 degree")
plt.ylabel("Flux Density(kJy)")
plt.xlabel("Degrees averaged over")
plt.legend()
plt.savefig("/arc/projects/chime_frb/mseth/plots/averaging_HAs_vs_flux")

pdb.set_trace()