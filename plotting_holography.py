import numpy as np 
import os
import matplotlib.pyplot as plt
import pdb
import copy
import scipy.signal 

'''
Takes .npz files for each frequency and combines everything into one .npz file with 
normalized response for xx pol, yy pol, and intensity. 

Plots the dynamic-spectrum-like beam response. 

'''
freqs = np.linspace(400.390625, 800, 1024) #1024 frequencies
has = np.linspace(-105, 104.90278, 2160)   #2160 HAs in holography data

# Load in all the .npz files & combine
holography_data = np.load('/arc/projects/chime_frb/mseth/Sorted_Normalized_holography_data.npz')
intensity = holography_data['intensity']


intensity_masked = intensity[:,1760:1840]

intensity_max = np.max(intensity_masked)
intensity_norm = intensity / intensity_max


f'''
#Unmasked beam response
plt.figure()
plt.plot(beam_response)
#plt.yscale('log')
plt.ylabel('Normalised sensitivity')
plt.xlabel('Frequency_bins')
#plt.savefig('/arc/projects/chime_frb/mseth/plots/masking_rfi_holography/nolog_beam_response')
'''


## MASKING? ##
ha_idx = 345
ha_idxs = np.arange(ha_idx-6, ha_idx+6)
beam_response = intensity_norm[0:512, ha_idx]

beam_response = intensity_norm[0:512, 340]
beam_response[beam_response==0] = np.nan
beam_copy = copy.deepcopy(beam_response)
peaks, properties = scipy.signal.find_peaks(beam_copy, prominence=0.0004, width=0.0001)
widths = properties['widths']

for peak, width in zip(peaks, widths):
    beam_slice = beam_copy[peak-20:peak-10]
    median = np.nanmedian(beam_slice)

    lower_ind = np.round(peak - 10* width).astype(int)
    upper_ind = np.round(peak + 10* width).astype(int)

    beam_copy[lower_ind:upper_ind] = median

peaks2 = scipy.signal.find_peaks(beam_copy)

beam_copy2 = copy.deepcopy(beam_copy)
difference = np.abs(beam_copy2 - np.nanmedian(beam_copy2))
std = np.nanstd(beam_copy2)

idxs = np.where(difference >= 1.5 * std)

for idx in idxs[0]:
    beam_copy2[idx] = np.nan

nanidxs = np.where(np.isnan(beam_copy2))
for nanidx in nanidxs[0]:
    beam_slice = beam_copy2[nanidx-10:nanidx+10]
    median = np.nanmedian(beam_slice)
    beam_copy2[nanidx] = median
                
plt.figure()
plt.plot(beam_response)
plt.plot(beam_copy2, color='r')
plt.yscale('log')
plt.savefig(f'/arc/projects/chime_frb/mseth/plots/masking_rfi_holography/340_log')
plt.close()

pdb.set_trace()


masked_beams = []
for i in ha_idxs:
    beam_response = intensity_norm[0:512, i]
    beam_response[beam_response==0] = np.nan
    beam_copy = copy.deepcopy(beam_response)
    peaks, properties = scipy.signal.find_peaks(beam_copy, prominence=0.0004, width=0.001)
    widths = properties['widths']

    for peak, width in zip(peaks, widths):
        beam_slice = beam_copy[peak-20:peak-10]
        median = np.nanmedian(beam_slice)

        lower_ind = np.round(peak - 5* width).astype(int)
        upper_ind = np.round(peak + 5* width).astype(int)

        beam_copy[lower_ind:upper_ind] = median

    peaks2 = scipy.signal.find_peaks(beam_copy)

    beam_copy2 = copy.deepcopy(beam_copy)
    difference = np.abs(beam_copy2 - np.nanmedian(beam_copy2))
    std = np.nanstd(beam_copy2)

    idxs = np.where(difference >= 1.5 * std)

    for idx in idxs[0]:
        beam_copy2[idx] = np.nan

    nanidxs = np.where(np.isnan(beam_copy2))
    for nanidx in nanidxs[0]:
        beam_slice = beam_copy2[nanidx-10:nanidx+10]
        median = np.nanmedian(beam_slice)
        beam_copy2[nanidx] = median
                
    plt.figure()
    plt.plot(beam_response)
    plt.plot(beam_copy2, color='r')
    plt.yscale('log')
    plt.savefig(f'/arc/projects/chime_frb/mseth/plots/masking_rfi_holography/{i}_masked')
    plt.close()
        
    masked_beams.append(beam_copy2)
    
stack = np.vstack(masked_beams)
averaged_beam = np.nanmean(stack, axis=0)
    
pdb.set_trace()    

    
#Masked beam response 
plt.figure()
plt.plot(beam_response)
plt.plot(averaged_beam, color='r')
plt.yscale('log')
plt.ylabel('Normalised sensitivity')
plt.xlabel('Frequency_bins')
plt.savefig('/arc/projects/chime_frb/mseth/plots/masking_rfi_holography/averaged')

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