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
Speak = 15e+3 #15kJy

bb_Speaks = []

'''
Modeling pulses
(Gaussian--intrinsic width convolved with exponential--scattering time)
'''
stokes_dt = 1e-3  #1ms time resolution
bb_dt = 2.56e-6 #2.56us time resolution
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

plt.figure(figsize=(10,5))
plt.semilogx(sigma, bb_Speaks/1000)
plt.xticks(
    [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
    ["100 ns", "1 μs", "10 μs", "100 μs", "1 ms"]
)
plt.xlabel("Intrinsic pulse width")
plt.ylabel("Peak Flux in Baseband (kJy)")
plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
plt.grid(True, which='both')
plt.savefig("15kJy_widthvsbb.png")



