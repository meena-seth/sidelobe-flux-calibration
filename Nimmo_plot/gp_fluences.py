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

dicts = ['Sallmen', 'Cordes', 'Hankins', 'Crossley', 'Jessner', 'Meyers', 'Bera', 'Sokolowski']
for dict in dicts[0:4]:
      dict['fluence'] = dict['flux'] * dict['width'] #Jy-s

plt.figure()
ax = plt.gca()

ax.set_xscale('log')
ax.set_xlim(0.1, 16)
xticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 16]
xticklabels = ["100 MHz", "200 MHz", "500 MHz","1 GHz", "2 GHz", "5 GHz", "10 GHz", "16 GHz"]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)



#### MY DATA 
data = np.load('/Users/meenaseth/sidelobe-flux-calibration/fluxcal_results.npz', allow_pickle=True)
fluences = data['fluences']
plt.scatter(600*10**6, fluences, label='This work')


#### Hankins 2003 
cnano=np.loadtxt('crab_nano.txt')
for n in range(len(cnano)):
    plotx, ploty = plot_point(cnano[n][0], cnano[n][1], freq=5.5*10**9)
    plt.scatter(plotx, ploty, color='orange', marker='+', label='Nanoshots')

#### Hankins 2003 



