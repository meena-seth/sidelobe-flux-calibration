import numpy as np 
import os
import matplotlib.pyplot as plt
import pdb

'''
Takes .npz files for each frequency and combines everything into one .npz file with 
normalized response for xx pol, yy pol, and intensity. 

Plots the dynamic-spectrum-like beam response. 

'''

# Load in all the .npz files & combine
path_to_npzs = "/arc/projects/chime_frb/mseth/calibration_data/holography_data"  

npz_files = []
for (root, dirs, file) in os.walk(path_to_npzs):
    for f in file: 
        npz_files.append(os.path.join(root, f))
        
        
# Need to sort npz_files
freq_names =[]
for file in npz_files:
    filename = file.split("/")
    freq_name = float(os.path.splitext(filename[7])[0])
    freq_names.append(freq_name)
        
freq_names.sort()

sorted_filepaths = []
for freq_name in freq_names:
    filename = f'{freq_name}.npz'
    filepath = os.path.join(path_to_npzs, filename)
    sorted_filepaths.append(filepath)

pdb.set_trace()

    
xx_list = []
yy_list = []
for filepath in sorted_filepaths:
    data = np.load(filepath)
    xx = data['XX']
    yy = data['YY']
    xx_list.append(xx)
    yy_list.append(yy)

xx = np.vstack(xx_list)   #[1024, 2160] [freq, HA]
yy = np.vstack(yy_list)   #[1024, 2160] [freq, HA]

# Normalizing & calculating intensity 
xx_masked =  xx[:,1040:1120]
xx_max = np.max(xx_masked)
xx_norm = xx / xx_max

yy_masked = yy[:,1040:1120]
yy_max = np.max(yy_masked)
yy_norm = yy / yy_max

intensity = xx + yy / 2
intensity_masked = intensity[:,1760:1840]
intensity_max = np.max(intensity_masked)
intensity_norm = intensity / intensity_max  #[1024, 2160] [freq, HA]


#Saving 
np.savez("/arc/projects/chime_frb/mseth/Sorted_Normalized_holography_data", xx_norm=xx_norm, yy_norm=yy_norm, intensity_norm=intensity_norm, intensity=intensity)

pdb.set_trace()
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

freqs = np.linspace(400.390625, 800, 1024) #1024 frequencies
has = np.linspace(-105, 104.90278, 2160)   #2160 HAs in holography #Plotting intensity
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

plt.savefig('Intensity_beam_response_sorted')