import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
print(mpl.__version__)
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.patches import Rectangle

def flux_to_luminosity(peak_flux):
	result = 4 * np.pi * np.square(6.171 * 10**19) * peak_flux * 10**(-19)
	return result

def luminosity_to_flux(peak_luminosity):
     result = peak_luminosity / (4 * np.pi * np.square((6.171 * 10**19)) * 10**(-19))
     return result




mpl.rcParams['font.size'] = 7
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 7
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['xtick.major.pad']='4'
mpl.rcParams['ytick.major.pad']='4'

#* NB. 1 W.Hz^{-1} == 1.05026*10^{-11} Jy.kpc^2    * 
#* --> L = T_B*y**2*(2.761e-23)    Watts/Hz        *      
#*      = [ ]*(1.05025e-13)  Jy,kpc^2              *
# note you need to multiply by 1e9**2 to convert to GHz 

x=np.linspace(1e-10,1e2,100)
plt.figure()
ax=plt.axes()

def plot_point(flux, width, freq):
     '''
     Flux in Jy, width in s, freq in Hz 
     '''
     lum = flux_to_luminosity(flux)
     lum /= 1e20 * 4 * np.pi
     freq_GHz = freq/1e+9
     scaled_width = width * freq_GHz
     return lum, scaled_width

#### MY DATA ####
data = np.load('/Users/meenaseth/sidelobe-flux-calibration/fluxcal_results.npz', allow_pickle=True)
lums = data['lums'] #ergs/s/Hz
lums /= 1e20 * 4*np.pi
widths = np.load('/Users/meenaseth/sidelobe-flux-calibration/pulse_widths.npz', allow_pickle=True)['widths'] #s
plt.scatter(widths,lums,color='orange',marker='x',alpha=0.7)
plt.text(5e-5,1e4,'This work',color='orange')

#### Cordes 2004 ####
my_x, my_y = plot_point(flux=155e+3, width=100e-6, freq=413e+6)
plt.scatter(my_x, my_y, color='r', marker='*')
plt.text(my_x, my_y*1.1, 'Cordes 2004')
print(my_x, my_y)

#### Bera & Chengalur 2019 ####
my_x, my_y = plot_point(flux=4e+6, width=1.1e-6, freq=1330e+6)
plt.scatter(my_x, my_y, color='r', marker='*')
plt.text(my_x, my_y*1.1, 'Bera & Chengalur 2019')
print(my_x, my_y)

#### Bera & Chengalur 2019 ####
my_x, my_y = plot_point(flux=4e+6, width=1.1e-6, freq=1330e+6)
plt.scatter(my_x, my_y, color='r', marker='*')
plt.text(my_x, my_y*1.1, 'Bera & Chengalur 2019')


#### Crab GPs from Karrupassmy 2010 ####
crabgrp=np.loadtxt('crab_giant.txt')
crabip=np.loadtxt('crab_giant.ip.txt')
crabgrpx=[]
crabgrpy=[]
for n in range(len(crabgrp)):
    crabgrpx.append(crabgrp[n][0]*crabgrp[n][1]*1e-3)
    crabgrpy.append(crabgrp[n][2]/crabgrp[n][1] * (2)**2)

for n in range(len(crabip)):
    crabgrpx.append(crabip[n][0]*crabip[n][1]*1e-3)
    crabgrpy.append(crabip[n][2]/crabip[n][1] * (2)**2)

plt.scatter(crabgrpx,crabgrpy,color='coral',marker='x',alpha=0.01)
plt.text(5e-6,1e0,'Crab GRPs',color='coral')

#### Crab Nanoshots (Hankins2003 and Jessner2010) ####
cnano=np.loadtxt('crab_nano.txt')
for n in range(len(cnano)):
    plt.scatter(cnano[n][0],cnano[n][1],color='orange',marker='+')
plt.text(2e-9,1e2,'Crab nanoshots',color='orange')

check_x, check_y = plot_point(8000, 2.2e-9, 5.5e+9)
plt.scatter(check_x, check_y, marker='*', color='k')


ax = plt.gca()


ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-9,10)
ax.set_ylim(1e-6,1e15)

ax.set_ylabel(r'Spectral Luminosity [erg s$^{-1}$ Hz$^{-1}$]')
ax.set_xlabel(r'Transient Duration ($\nu\ W$) [GHz s]')

# ticks
ax.tick_params(axis = 'both', which = 'minor', labelsize = 0, labelcolor='white')
ax.set_xticks([1e-8,1e-6,1e-4,1e-2,1], minor=True)
ax.set_yticks([1e-3/(4*np.pi),1e-2/(4*np.pi),1/(4*np.pi),10/(4*np.pi),1e3/(4*np.pi),1e4/(4*np.pi),1e6/(4*np.pi),1e7/(4*np.pi),1e9/(4*np.pi),1e10/(4*np.pi),1e12/(4*np.pi),1e13/(4*np.pi),1e15/(4*np.pi)],minor=True)#,1e16,1e18], minor=True)

ax.set_xticks([1e-9,1e-7,1e-5,1e-3,1e-1])
ax.set_xticklabels([r'$10^{-9}$',r'$10^{-7}$',r'$10^{-5}$',r'$10^{-3}$',r'$10^{-1}$'])

ax.set_yticks([1e-4/(4*np.pi),1e-1/(4*np.pi),1e2/(4*np.pi),1e5/(4*np.pi),1e8/(4*np.pi),1e11/(4*np.pi),1e14/(4*np.pi)])#,1e17])
ax.set_yticklabels([r'$10^{16}$',r'$10^{19}$',r'$10^{22}$',r'$10^{25}$',r'$10^{28}$',r'$10^{31}$',r'$10^{34}$'])#,r'$10^{37}$'])



plt.savefig('/Users/meenaseth/sidelobe-flux-calibration/figure3_KN.png',format='png',dpi=300)
plt.show()
