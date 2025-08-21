import numpy as np 
import os
import matplotlib.pyplot as plt
import pdb

'''
Takes .npz files for each frequency and combines everything into one .npz file with 
normalized response for xx pol, yy pol, and intensity. 

Plots the dynamic-spectrum-like beam response. 

'''
freqs = np.linspace(400.390625, 800, 1024) #1024 frequencies
has = np.linspace(-105, 104.90278, 2160)   #2160 HAs in holography data

# Load in all the .npz files & combine
holography_data = np.load('/arc/projects/chime_frb/mseth/nrao/Holography_data_2.npz')
intensity = holography_data['intensity']


intensity_masked = intensity[:,1760:1840]

intensity_max = np.max(intensity_masked)
intensity_norm = intensity / intensity_max

#intensity_max = np.max(intensity_masked, axis=1)
#intensity_norm = intensity / intensity_max[:, np.newaxis]  #[1024, 2160] [freq, HA]

pdb.set_trace()

#Plotting intensity
plt.figure()
fig, ax = plt.subplot_mosaic(
    '''
    AA
    ''',
    figsize = (6,6),
    layout = 'constrained')

pcm = ax['A'].pcolormesh(has, freqs, np.log10(intensity_norm))
ax['A'].set_ylabel('Frequencies (MHz)')
ax['A'].set_xlabel('HA')
ax['A'].set_title('Intensity')

fig.colorbar(pcm, ax=ax['A'])

plt.savefig('Intensity_beam_response_flatno')