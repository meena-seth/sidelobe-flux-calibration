import numpy as np 
import os
import matplotlib.pyplot as plt

'''
Takes .npz files for each frequency and combines everything into one .npz file with 
normalized response for xx pol, yy pol, and intensity. 

Plots the dynamic-spectrum-like beam response. 

'''

# Load in all the .npz files & combine
import sys
path_to_npzs = sys.argv[1]

npz_files = []
for (root, dirs, file) in os.walk(path_to_npzs):
    for f in file: 
        npz_files.append(os.path.join(root, f))
        
xx_list = []
yy_list = []
#pull out only the fn
fns = [os.path.basename(f) for f in npz_files]
#remove the .npz extension
freqs = [float(fn[:-4]) for fn in fns]
sort_indices = np.argsort(freqs)
freqs = np.array(freqs)[sort_indices]
npz_files = np.array(npz_files)[sort_indices]

#sort the npz files by frequency index
for file in npz_files:
    print(file)
    data = np.load(file)
    xx = data['XX']
    yy = data['YY']
    xx_list.append(xx)
    yy_list.append(yy)

xx = np.vstack(xx_list)   #[1024, 2160] [freq, HA]
yy = np.vstack(yy_list)   #[1024, 2160] [freq, HA]

# Normalizing & calculating intensity
import pdb; pdb.set_trace()
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
np.savez("Normalized_holography_data", xx_norm=xx_norm, yy_norm=yy_norm, intensity_norm=intensity_norm)

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

plt.savefig('XX_YY_beam_response')

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

plt.savefig('Intensity_beam_response')
