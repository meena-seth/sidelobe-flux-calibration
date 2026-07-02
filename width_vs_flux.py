import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import exponnorm
import matplotlib.ticker as ticker


'''
Pulse parameters
'''

sigma = np.linspace(80e-9, 2e-3, 8000) #Intrinsic widths
mu = 0
tau = 26e-6 #Scattering time, Nadeu 2026 p.10
Speak = 5e+3 #5kJy

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

for test_sigma in sigma:
    stokes = exponnorm.pdf(t, K=tau/test_sigma, loc=mu, scale=test_sigma)
    stokes *= Speak/stokes.max() 
    stokes_area = np.trapezoid(stokes,t)

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
plt.savefig("Widths_vs_BBflux.png")



split = 50e-6  # 50 microseconds

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(6, 8),
    sharey=False,
    constrained_layout=True
)

# Top subplot: <= 50 us
mask1 = sigma <= split
ax1.semilogx(sigma[mask1], bb_Speaks[mask1]/1000)
ax1.set_xlim(sigma.min(), split)
ax1.set_xticks([1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
ax1.set_xticklabels(["1 ns", "10 ns", "100 ns", "1 μs", "10 μs"])
ax1.set_ylabel("Peak Flux in Baseband (kJy)")
ax1.set_title("Intrinsic pulse widths ≤ 50 μs")

# Bottom subplot: >= 50 us
mask2 = sigma >= split
ax2.semilogx(sigma[mask2], bb_Speaks[mask2]/1000)
ax2.set_xlim(split, sigma.max())
ax2.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
ax2.set_xticklabels(["100 μs", "1 ms", "10 ms", "100 ms"])
ax2.set_xlabel("Intrinsic pulse width")
ax2.set_ylabel("Peak Flux in Baseband (kJy)")
ax2.set_title("Intrinsic pulse widths ≥ 50 μs")

#plt.savefig("Widths_vs_BBflux_subplots.png", dpi=200)

