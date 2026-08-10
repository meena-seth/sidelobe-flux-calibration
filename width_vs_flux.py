import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import exponnorm
from scipy import integrate
import matplotlib.ticker as ticker

manual=False
'''
Pulse parameters
'''

sigma = np.linspace(80e-9, 2e-3, 8000) #Intrinsic widths
mu = 0
tau = 26e-6 #Scattering time, Nadeu 2026 p.10
Speak = 5e+3 
Speak2 = 20e+3
scale_factor = Speak2/Speak

bb_Speaks = []

'''
Modeling pulses
(Gaussian--intrinsic width convolved with exponential--scattering time)
'''
stokes_dt = 1e-3  #1ms time resolution
bb_dt = 2.56e-6 #2.56us time resolution for CHIME bb
factor = int(stokes_dt/bb_dt) #~390

t = np.arange(-10e-3, 10e-3, stokes_dt) 
t_bb = np.arange(-10e-3, 10e-3, bb_dt) 

'''
Trying again (?)
'''
def gaussian(x, sigma, mu=0):
    '''
    x: array (time)
    '''
    return np.exp(-1 * 1/2 * np.square((x-mu)/sigma)) * 1/(sigma * np.sqrt(2*np.pi))

def exponential(x, tau=26e-6):
    '''
    x: array (time)
    tau: scattering timescale 
    '''
    return np.exp(-1 * x / tau)

def make_convolved(t, sig):
    return np.convolve(gaussian(x=t, sigma=sig), exponential(x=t))
    

for test_sigma in sigma:
    if manual:
        stokes = make_convolved(t, test_sigma)
        stokes *= Speak/stokes.max() 
        stokes_area = integrate.simpson(stokes)

        bb = make_convolved(t_bb, test_sigma)
        bb_norm = bb/bb.max()
        Speak_bb = stokes_area/integrate.simpson(bb_norm)
        bb_Speaks.append(Speak_bb)
    else:
        exp = -1/tau
        exp_stokes = t/tau
        stokes = exponnorm.pdf(t, K=tau/test_sigma, loc=mu, scale=test_sigma)
        stokes *= Speak/stokes.max() 
        stokes_area = np.trapezoid(stokes,t)

        exp_bb = t_bb/tau
        bb = exponnorm.pdf(t_bb, K=tau/test_sigma, loc=mu, scale=test_sigma)
        bb_norm = bb / bb.max() #Normalized shape for baseband, peak=1
        Speak_bb = stokes_area / np.trapezoid(bb_norm, t_bb) #Flux needed to match fluence
        bb_Speaks.append(Speak_bb)



bb_Speaks = np.array(bb_Speaks)

fig, ax_left = plt.subplots(figsize=(12, 5))
ax_left.semilogx(sigma, bb_Speaks / 1000, color='tab:blue')
ax_left.tick_params(labelsize=12)
ax_left.set_xticks(
    [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
    ["100 ns", "1 μs", "10 μs", "100 μs", "1 ms"],
    fontsize=12
)
ax_left.set_xlabel("Intrinsic pulse width", fontsize=15)
ax_left.set_ylabel("Peak Flux in Baseband (kJy)\n for 5kJy pulse in Stokes", fontsize=15)
ax_left.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax_left.grid(True, which='both')

# Right axis shows same relationship but values for a 20kJY pulse
ax_right = ax_left.twinx()
ax_right.tick_params(labelsize=12)
ax_right.set_ylabel("for 20kJy pulse in Stokes", fontsize=15)
def sync_right_axis(ax_left):
    y_min, y_max = ax_left.get_ylim()
    ax_right.set_ylim(y_min * scale_factor, y_max * scale_factor)

ax_left.callbacks.connect("ylim_changed", sync_right_axis)
sync_right_axis(ax_left)  

fig.tight_layout()
plt.savefig("widthvsbb.pdf", format='pdf', bbox_inches='tight')



