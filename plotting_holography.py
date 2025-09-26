import numpy as np 
import os
import matplotlib.pyplot as plt
import pdb

freqs = np.linspace(400.390625, 800, 1024) #1024 frequencies
has = np.linspace(-105, 104.90278, 2160)   #2160 HAs in holography 

## LOADING IN DATA 
path_to_holography = '/arc/projects/chime_frb/mseth/Sorted_Normalized_holography_data.npz'

holography = np.load(path_to_holography, allow_pickle=True)

intensity_norm = holography['intensity_norm']
yy_norm = holography['yy_norm']
xx_norm = holography['xx_norm']

plt.subplots(figsize=(13, 4))
plt.plot(has, intensity_norm[809])
plt.yscale('log')
plt.xlabel('HA (deg)')
plt.ylabel('Normalized sensitivity')
plt.title('Intensity, 716.41 MHz')
#plt.savefig('/arc/projects/chime_frb/mseth/plots/holography_example.pdf')

## PLOTTING 
# 




#Plotting xx and yy 
fig, ax = plt.subplot_mosaic(
    '''
    AA
    BB
    ''',
    figsize = (8,10),
    layout = 'constrained')

pcm_xx = ax['A'].pcolormesh(has, freqs, np.log10(xx_norm))
ax['A'].set_ylabel('Frequencies (MHz)')
ax['A'].set_xlabel('HA')
ax['A'].set_title('XX Polarization')

pcm_yy = ax['B'].pcolormesh(has, freqs, np.log10(yy_norm))
ax['B'].set_ylabel('Frequencies (MHz)')
ax['B'].set_xlabel('HA')
ax['B'].set_title('YY Polarization')

fig.colorbar(pcm_xx, ax=ax['A'])
fig.colorbar(pcm_yy, ax=ax['B'])
plt.suptitle('CHIME Primary beam response')
#plt.savefig('XX_YY_beam_response')



#Plotting intensity
plt.figure()
fig, ax = plt.subplot_mosaic(
    '''
    AA
    ''',
    figsize = (7,7),
    layout = 'constrained')

pcm = ax['A'].pcolormesh(has, freqs, np.log10(intensity_norm))
ax['A'].tick_params(axis='both', labelsize=14)
ax['A'].set_ylabel('Frequencies (MHz)', fontsize=15)
ax['A'].set_xlabel('HA', fontsize=15)
ax['A'].set_title('Intensity', fontsize=19)
cbar = fig.colorbar(pcm, ax=ax['A'])
cbar.ax.tick_params(labelsize=14)

plt.savefig("/arc/projects/chime_frb/mseth/plots/fixed_axes_labels/3_chime_beam_response.pdf")

