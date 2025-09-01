import numpy as np 
import matplotlib.pyplot as plt
import pdb
from matplotlib.lines import Line2D

def snr_radiometer(v, T_sys, Gain, Flux, distance):
    '''
    Good & Andersen et. al 2021
    Gives SNR assuming 1ms wide pulse & 2 polarizations
    '''
    end_flux = Flux * (4 / (distance**2))
    S_sys = T_sys / Gain
    result = 0.868 * np.sqrt(2 * 0.001 * v*10**6) * (end_flux / S_sys)
    
    return result 


# Values from our data + Andersen 2023 (CHIME band)
chime_tsys = 50 #K
chime_gain = 1.2 # K Jy^-1, Andersen 2023 
chime_v = 400 #Mhz 

# Values from Stovall 2014, GB Northern Celestial Cap Survey (centered around 350 MHz)
gbt_tsys = 23 #K
gbt_gain = 2 #K Jy^-1
gbt_v = 100 #MHz

# Values from Chuan-Peng Zhang et. al 2023 for UWB-1 Receiver (0-1100 MHz)
fast_tsys = 24 
fast_gain = (14.3 + 11.3)/2 
fast_v = 1100

# Values from Bales & Jameson 2020 for L band 
meerk_tsys = 18 
meerk_gain = 2.8
meerk_v = 1711 - 856 #de Villiers & Mattieu 2023

# Values from Das & Roy 2025 for Band-3
gmrt_tsys = 130 
gmrt_gain = 0.35 * (14+22)
gmrt_v = 200 

parkes_tsys = 21
parkes_gain = 1.8
parkes_v = 300 


start_fluxes = np.linspace(10000, 85*10**3, 100)
#10**4, 8*10**4, 100)
distances = np.linspace(700, 3900, 100) #Kpc 

rows = len(distances)
cols = len(start_fluxes)
snr_chime = np.empty((rows, cols), dtype=float)
snr_gbt = np.empty((rows, cols), dtype=float)
snr_fast = np.empty((rows, cols), dtype=float)
snr_meerk = np.empty((rows, cols), dtype=float)
snr_gmrt = np.empty((rows, cols), dtype=float)
snr_parkes = np.empty((rows, cols), dtype=float)


for i in range(rows):
    for j in range(cols):
        snr = snr_radiometer(v=meerk_v,
                                   T_sys=meerk_tsys,
                                   Gain=meerk_gain,
                                   distance=distances[i],
                                   Flux=start_fluxes[j]
                                  )
        snr_meerk[i][j] = snr
        continue
        
for i in range(rows):
    for j in range(cols):
        snr = snr_radiometer(v=gmrt_v,
                                   T_sys=gmrt_tsys,
                                   Gain=gmrt_gain,
                                   distance=distances[i],
                                   Flux=start_fluxes[j]
                                  )
        snr_gmrt[i][j] = snr
        continue
        
for i in range(rows):
    for j in range(cols):
        snr = snr_radiometer(v=parkes_v,
                                   T_sys=parkes_tsys,
                                   Gain=parkes_gain,
                                   distance=distances[i],
                                   Flux=start_fluxes[j]
                                  )
        snr_parkes[i][j] = snr
        continue
        
        
for i in range(rows):
    for j in range(cols):
        snr = snr_radiometer(v=chime_v,
                                   T_sys=chime_tsys,
                                   Gain=chime_gain,
                                   distance=distances[i],
                                   Flux=start_fluxes[j]
                                  )
        snr_chime[i][j] = snr
        continue
        
        
for i in range(rows):
    for j in range(cols):
        snr = snr_radiometer(v=gbt_v,
                                   T_sys=gbt_tsys,
                                   Gain=gbt_gain,
                                   distance=distances[i],
                                   Flux=start_fluxes[j]
                                  )
        snr_gbt[i][j] = snr
        continue
        
for i in range(rows):
    for j in range(cols):
        snr = snr_radiometer(v=fast_v,
                                   T_sys=fast_tsys,
                                   Gain=fast_gain,
                                   distance=distances[i],
                                   Flux=start_fluxes[j]
                                  )
        snr_fast[i][j] = snr
        continue
    
X, Y = np.meshgrid(distances, start_fluxes/1e3)

plt.figure()
chime = plt.contour(X, Y*4, snr_chime.T, levels=[6], colors='lightcoral', label="CHIME")
gbt = plt.contour(X, Y*4, snr_gbt.T, levels=[6], colors='orangered', label="GBT")
fast = plt.contour(X, Y*4, snr_fast.T, levels=[6], colors='yellowgreen', label="FAST")
meerk = plt.contour(X, Y*4, snr_meerk.T, levels=[6], colors='lightseagreen', label="MEERKAT")
gmrt = plt.contour(X, Y*4, snr_gmrt.T, levels=[6], colors='rebeccapurple', label="GMRT")
parkes = plt.contour(X, Y*4, snr_parkes.T, levels=[6], colors='mediumvioletred', label="PARKES")

legend_elements = [
    Line2D([0], [0], color='lightcoral', lw=1, label='CHIME'),
    Line2D([0], [0], color='orangered', lw=1, label='GBT'),
    Line2D([0], [0], color='yellowgreen', lw=1, label='FAST'),
    Line2D([0], [0], color='lightseagreen', lw=1, label='MEERKAT'),
    Line2D([0], [0], color='rebeccapurple', lw=1, label='GMRT'),
    Line2D([0], [0], color='mediumvioletred', lw=1, label='PARKES'),

]

plt.legend(handles=legend_elements, loc='best')


    
plt.axhline(y=(4* 80.8), linestyle='dotted', color='gray') 
#plt.text(1.2*10**3, 4 * 78, 'Brightest pulse in our sample', rotation='horizontal')

plt.axhline(y=(4 * 13.6), linestyle='dotted', color='gray')
#plt.text(3.7*10**3, 4*14, 'Lowest luminosity pulse in our sample', rotation='horizontal', ha='left', va='bottom', wrap=True)

plt.axhline(y=(4 * 51.8), linestyle='dotted', color='gray')
#plt.text(1.2*10**3, 52, 'Brightest pulse in our sample (within 90 deg of main beam)', rotation='horizontal', ha='left', va='bottom')



plt.axvline(x=778, linestyle='--')
plt.text(790, 66, 'Andromeda', rotation='vertical', ha='left', va='bottom')

plt.axvline(x=900, linestyle='--')
plt.text(910, 66, 'M33', rotation='vertical', ha='left', va='bottom')

plt.axvline(x=1700, linestyle='--')
plt.text(1750, 66, 'Extent of Local Group', rotation='vertical', ha='left', va='bottom')

plt.axvline(x=3600, linestyle='--')
plt.text(3690, 66, 'M81/FRB20200120E', rotation='vertical', ha='left', va='bottom')

#plt.legend()
plt.xlabel('Distance (kpc)')
plt.ylabel(f'Pseudo-Luminosity (kJy kpc^2)')
plt.title('Detection Threshold for S/N=6')
plt.xscale('log')
plt.savefig("/arc/projects/chime_frb/mseth/plots/distance_thresholds/thresholds.png")
pdb.set_trace()
