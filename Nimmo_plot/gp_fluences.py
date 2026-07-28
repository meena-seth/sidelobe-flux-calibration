import numpy as np
import matplotlib.pyplot as plt

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
    "width": [22e-6, 300e-6, 0.11e-3] # s
}

Cordes = {
    "obs": ["Arecibo", "Arecibo", "Arecibo", "Arecibo"],
    "freq": [0.43, 1.475, 2.33, 2.85],   # GHz
    "flux": [155000, 1030, 86, 89],      # Jy
    "width": [100e-6, 100e-6, 100e-6, 100e-6]  # Estimated
}

Hankins = {
    "obs": ["Arecibo"],
    "freq": [5.5],      # Not specified whether 5.5Ghz or 8.6GHz, just choose 5.5 for now
    "flux": [1000],      # Jy
    "width": [2e-9]      # s
}

Crossley = {
    "obs": ["VLA"] * 8,
    "freq": [0.33, 0.333, 1.34, 1.34, 1.69, 1.69, 4.765, 4.765],  # GHz
    "flux": [1500, 200, 600, 80000, 2200, 41000, 1000, 120000],   # Jy
    "width": [400e-6, 400e-6, 19e-6, 1.5e-6, 5e-6, 1.1e-6, 1.5e-6, 0.2e-6]
}

Meyers = {
    "obs": ["MWA", "MWA", "MWA", "MWA", "Parkes", "Parkes"],
    "freq": [0.12096, 0.16576, 0.18496, 0.21056, 0.732, 3.1],  # GHz
    "fluence": [20.42, 19.96, 9.54, 7.89, 5.77, 0.077]         # Jy s
}

Jessner = {
    "obs": ["Effelsberg 100m", "Effelsberg 100m"],
    "freq": [8.5, 15.1],     # GHz
    "flux": [150000, 60000], # Jy
    "width": [100e-6, 100e-6]    # Not specified but says envelopes tend to be ~100us, so use that.
}

Bera = {
    "obs": ["NCRA 15m"],
    "freq": [1.33],          # GHz
    "fluence": [4.7]         # Jy s
}

Sokolowski = {
    "obs": ["SKA Low"],
    "freq": [0.215],         # GHz
    "fluence": [76e-3]          # Jy s
}

dict_names = ['Sallmen 1999', 'Cordes 2004', 'Hankins 2003', 'Crossley 2010', 'Jessner 2018', 'Meyers 2017', 'Bera 2019', 'Sokolowski 2025']
dicts = [Sallmen, Cordes, Hankins, Crossley, Jessner, Meyers, Bera, Sokolowski]
for dict in dicts[0:5]:
      dict['fluence'] = np.array(dict['flux']) * np.array(dict['width']) #Jy-s

plt.figure(figsize=(8, 5))
ax = plt.gca()

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.1, 16)
xticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 16] #GHz
xticklabels = ["0.1", "0.2", "0.5","1", "2", "5", "10", "16"] #GHz
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

for dict, name in zip(dicts, dict_names):
      plt.scatter(np.array(dict['freq']), np.array(dict['fluence']), label=name, marker='+')

#### MY DATA 
data = np.load('/Users/meenaseth/sidelobe-flux-calibration/fluxcal_results.npz', allow_pickle=True)
widths = np.load('/Users/meenaseth/sidelobe-flux-calibration/pulse_widths.npz', allow_pickle=True)['widths'] #s
fluences = data['fluences'] #Jy-s 
yvals = np.max(fluences), np.min(fluences), np.median(fluences)
xvals = np.full_like(yvals, 0.6)
plt.scatter(xvals, yvals, label='This work', marker='*') #Put everything at 600MHz

plt.legend(fontsize=6)
plt.grid('show', alpha=0.4)
plt.xlabel('Central Observing Frequency (GHz)')
plt.ylabel('Fluence (Jy-s)')
plt.savefig('GP_fluences.png', dpi=800)

