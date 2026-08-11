import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import pdb


def flux_to_luminosity(peak_flux):
	result = 4 * np.pi * np.square(6.171 * 10**19) * peak_flux * 10**(-19)
	return result

def luminosity_to_flux(peak_luminosity):
     result = peak_luminosity / (4 * np.pi * np.square((6.171 * 10**19)) * 10**(-19))
     return result

def plot_point(xval, yval, freq):
     '''
      xval - Ghz s 
      yval - luminosity
      freq - Hz
     '''
     flux = luminosity_to_flux(yval)
     width = xval / (freq/10**9) #s
     fluence = width * flux 
     return freq, fluence

Sallmen = {
    "obs": ["VLA", "VLA", "GBO 25m"],
    "freq": [1.4, 1.4, 0.6],          # GHz
    "flux": [3000, 3400, 7000],       # Jy
    "width": [22e-6, 300e-6, 0.11e-3], # s
    "estimate":True,
}

Cordes = {
    "obs": ["Arecibo", "Arecibo", "Arecibo", "Arecibo"],
    "freq": [0.43, 1.475, 2.33, 2.85],   # GHz
    "flux": [155000, 1030, 86, 89],      # Jy
    "width": [100e-6, 100e-6, 100e-6, 100e-6],  # Estimated
    "estimate":True
}

Hankins2003 = {
    "obs": ["Arecibo"],
    "freq": [5.5],      # Not specified whether 5.5Ghz or 8.6GHz, just choose 5.5 for now
    "flux": [1000],      # Jy
    "width": [2e-9],      # s
    "estimate":False
}

Hankins2007 = {
    "obs": ["Arecibo"],
    "freq": [9.25],      # Not specified whether 5.5Ghz or 8.6GHz, just choose 5.5 for now
    "flux": [2e+6],      # Jy
    "width": [0.4e-9],      # s
    "estimate":False
}

Bhat = {
    "obs": ["MWA"],
    "freq": [0.200],         # GHz
    "fluence": [185],          # Jy s
    "estimate":False
}

Popov = {
    "obs": ["Kalyazin"],
    "freq": [0.6, 1.4],         # GHz
    "fluence": [0.21, 1.48],          # Jy s
    "estimate":True
}

Crossley = {
    "obs": ["VLA"] * 8,
    "freq": [0.33, 0.333, 1.34, 1.34, 1.69, 1.69, 4.765, 4.765],  # GHz
    "flux": [1500, 200, 600, 80000, 2200, 41000, 1000, 120000],   # Jy
    "width": [400e-6, 400e-6, 19e-6, 1.5e-6, 5e-6, 1.1e-6, 1.5e-6, 0.2e-6],
    "estimate":True
}

Meyers = {
    "obs": ["MWA", "MWA", "MWA", "MWA", "Parkes", "Parkes"],
    "freq": [0.12096, 0.16576, 0.18496, 0.21056, 0.732, 3.1],  # GHz
    "fluence": [20.42, 19.96, 9.54, 7.89, 5.77, 0.077],         # Jy s
    "estimate":True
}

Jessner = {
    "obs": ["Effelsberg 100m", "Effelsberg 100m"],
    "freq": [8.5, 15.1],     # GHz
    "flux": [150000, 60000], # Jy
    "width": [100e-6, 100e-6],    # Not specified but says envelopes tend to be ~100us, so use that.
    "estimate":True
}

Bera = {
    "obs": ["NCRA 15m"],
    "freq": [1.33],          # GHz
    "fluence": [4.7],         # Jy s
    "estimate":False
}

Thulasiram = {
    "obs": ["Algonquin"],
    "freq": [0.5, 0.575, 0.713, 0.563, 0.602, 0.528],          # GHz
    "fluence": [0.285, 0.146, 0.052, 0.069, 0.050, 0.071],         # Jy s
    "estimate":False
}

Sokolowski = {
    "obs": ["SKA Low"],
    "freq": [0.215],         # GHz
    "fluence": [76e-3],          # Jy s
    "estimate":False
}


### NON-CRAB SOURCES 

J2109_50 = {
    "obs": ["CHIME"],
    "freq": [0.6],         # GHz
    "fluence": [20],          # Jy s
    "estimate":False
}

B0329_54 = {
    "obs": ["GMRT"],
    "freq": [1.41],              # GHz
    "fluence": [4.8e-3],          # Jy s - from Kramer 2003, brightest at 1.41GHz ~800mJy, w~6ms
    "estimate":True
}

Vela = {
    "obs": ["Parkes"],
    "freq": [1.41],         # GHz
    "fluence": [0.125],          # Jy s - From Johnston 2001
    "estimate":True
}

sgr1935 = {
    "obs": ["CHIME"], #CHIME collaboration 2020
    "freq": [0.3, 0.7, 1.374],         # GHz. Third entry is from Bochenek 2020 showing FRB like burst.
    "fluence": [480, 220,1.5e+3],          # Jy s
    "estimate":False
}


crab_dicts = [Sallmen, Hankins2003, Cordes, Hankins2007, Bhat, Popov, Crossley, Jessner, Meyers, Bera, Sokolowski]
crab_names = ['Sallmen 1999', 'Hankins 2003', 'Cordes 2004','Hankins 2007', 'Bhat 2007', 'Popov 2009', 
              'Crossley 2010', 'Jessner 2010', 'Meyers 2017', 'Bera 2019', 'Sokolowski 2025']

nanoshots = ['Hankins 2003', 'Hankin 2007', 'Jessner 2010']
microbursts = ['Sallmen 1999', 'Crossley 2010', 'Bera 2019', 'Bhat 2007', 'Popov 2009']

other_dicts = J2109_50, B0329_54, Vela, sgr1935
other_names = ['J2019+50', 'B0329+54', 'Vela Pulsar', 'SGR1935+15']

for dict in crab_dicts:
        try:
              dict['fluence'] = np.array(dict['flux']) * np.array(dict['width']) #Jy-s
        except Exception as e:
              continue

plt.figure(figsize=(13, 6))
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.1, 16)
xticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 16] #GHz
xticklabels = ["0.1", "0.2", "0.5","1", "2", "5", "10", "16"] #GHz
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

for dict, name in zip(crab_dicts, crab_names):
    if name in nanoshots:
         marker = 'P'
    elif name in microbursts:
         marker = 'X'
    else:
         marker = 'o'

    if dict.get('estimate', True):
         plt.scatter(np.array(dict['freq']), np.array(dict['fluence']), marker=marker, facecolor='none', edgecolors='gray', linewidths=1)
    else:
         plt.scatter(np.array(dict['freq']), np.array(dict['fluence']), marker=marker, facecolor=to_rgba('gray', alpha=0.8), edgecolors='gray', linewidths=1)

colors = ['orangered', 'seagreen', 'royalblue', 'mediumvioletred']
for dict, name, color in zip(other_dicts, other_names, colors):
    if dict.get('estimate', True):
         plt.scatter(np.array(dict['freq']), np.array(dict['fluence']), marker='o', facecolor='none', edgecolors=color, linewidths=1)
    else:
         plt.scatter(np.array(dict['freq']), np.array(dict['fluence']), marker='o', facecolor=to_rgba(color, alpha=0.8), edgecolors=color, linewidths=1)


legend1_markers = ['o', 'X', 'P']
legend1_labels = ['GP envelope', 'Microburst', 'Nanoshot']
legend1_elements = [Line2D([0], [0], linestyle='none', marker=marker, markerfacecolor=to_rgba('gray', alpha=0.8), markeredgecolor='gray', markeredgewidth=1, label=label)
                    for marker, label in zip(legend1_markers, legend1_labels)
                   ]
legend1_elements.append(Line2D([0], [0], linestyle='none', marker='', label='(Unfilled markers indicate \nestimated values)'))

legend2_elements = [Line2D([0], [0], color=color, lw=2, label=label)
                    for color, label in zip(colors, other_names)
                   ]


legend1 = ax.legend(handles=legend1_elements, title='Crab Pulses', loc='lower left')
legend1.get_title().set_ha('left')
ax.add_artist(legend1)
legend2 = ax.legend(handles=legend2_elements, title='Other Sources', loc='lower right')
legend2.get_title().set_ha('left')

         

#### MY DATA 
data = np.load('/Users/meenaseth/sidelobe-flux-calibration/fluxcal_results.npz', allow_pickle=True)
widths = np.load('/Users/meenaseth/sidelobe-flux-calibration/pulse_widths.npz', allow_pickle=True)['widths'] #s
fluences = data['fluences'] #Jy-s 
yvals = np.max(fluences), np.min(fluences), np.median(fluences)
xvals = np.full_like(yvals, 0.6)
plt.scatter(xvals, yvals, label='This work', marker='o', facecolor=to_rgba('k', alpha=0.8), edgecolors= 'k',) #Put everything at 600MHz

#plt.legend(fontsize=6)
plt.grid('show', alpha=0.4)
plt.xlabel('Central Observing Frequency (GHz)')
plt.ylabel('Fluence (Jy-s)')
plt.savefig('GP_fluences.png', dpi=800, bbox_inches='tight')

