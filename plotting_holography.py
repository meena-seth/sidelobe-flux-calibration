import numpy as np 
import os
import matplotlib.pyplot as plt
import pdb
import copy
import scipy.ndimage
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

## Loading unmasked beam response 
beam_response = intensity_norm[0:512, 345]
beam_response[beam_response==0] = np.nan
#beam = beam_response[100:200]

#Unmasked beam response
plt.figure()
plt.plot(beam_response)
plt.yscale('log')
plt.ylabel('Normalised sensitivity')
plt.xlabel('Frequency_bins')
#plt.savefig('/arc/projects/chime_frb/mseth/plots/masking_rfi_holography/beam[100:200]')


#Trying to mask beam response? 
starts = np.array([0, 100, 200, 300, 400])
ends = np.array([100, 200, 300, 400, 512])

beam_copy = copy.deepcopy(beam_response)
beam_slices = []


#Unmasked beam response
plt.figure()
plt.plot(beam_response)
#plt.plot(beam_masked, color='r')
plt.yscale('log')
plt.ylabel('Normalised sensitivity')
plt.xlabel('Frequency_bins')
#plt.savefig('/arc/projects/chime_frb/mseth/plots/masking_rfi_holography/median_filter')


def mask_outliers(array):
    for i, j in zip(starts, ends):
        beam = array[i:j]
        difference = np.abs(beam - np.nanmedian(beam))
        std = np.nanstd(beam)
        beam[difference > 2*std] = np.nanmedian(beam)
        beam_slices.append(beam)
    beam_masked = np.hstack(beam_slices)
    return beam_masked 

beam_masked = mask_outliers(beam_copy)

f'''
for _ in range(5):
    current = mask_outliers(current)
    
pdb.set_trace()
'''

#Masked beam response 
plt.figure()
plt.plot(beam_response)
plt.plot(beam_masked, color='r')
plt.yscale('log')
plt.ylabel('Normalised sensitivity')
plt.xlabel('Frequency_bins')
plt.savefig('/arc/projects/chime_frb/mseth/plots/masking_rfi_holography/masked')

pdb.set_trace()



#For one 
median = np.nanmedian(beam)
std = np.nanstd(beam)
difference = np.abs(beam - median)
beam[difference >= 2*std] = median 


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