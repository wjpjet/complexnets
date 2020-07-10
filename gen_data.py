#from cusignal import chirp, spectrogram
import numpy as np
# Bandwidt
from numpy import random
from LFMSignal import *
from scipy.signal import chirp, spectrogram, hilbert

# What sampling rate should I use?

"""
Find notes I took from last Friday and write here


**************************************************"
*
*
*
*
*
*
*
*



"""



# Signal properties
fc_range = [10e9, 10e9]
bw_range = [4e6, 6e6]
pw_range = [24e-6, 64e-6]
# Nyquist Sampling for all signals
nyquist_scale = 1
fs = (max(fc_range) + max(bw_range)) * 2 * nyquist_scale
print("Sampling at: {} Hz".format(fs))
print()
# Sampling
#fc = np.random.randint(fc_range[0], fc_range[1], size=[1])
fc = 10e9
fs = 10e9
bw = np.random.randint(bw_range[0], bw_range[1], size=[1]) // 2
pw = (pw_range[1] - pw_range[0])*random.random_sample() + pw_range[0]# scale to [a, b) = (b-a)*pw + a

#pw = np.random.randint(pw_range[0], pw_range[1], size=[1])
f0 = fc - bw
f1 = fc + bw
#voltage = 100

print(" ".join(map(str, [fc, bw, pw])))
print(f0)
print(f1)

# Pulse width
seconds = pw
print(seconds)
#t = np.linspace(0, seconds, seconds * fs, endpoint=False)
lfm = LFM_Waveform(fs, f0, f1, seconds, voltage=1)
#lfm.data = hilbert(lfm.data)
print(lfm)
lfm.plot_wvfm()



# f, t, Sxx = spectrogram(lfm.data, fs)
# plt.pcolormesh(t, f, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.title('Waveform sampled at {} Hz'.format(fs))
# plt.show()


#lfm.spectrogram(fs=bw)
#energy_spectral_density(lfm.data, "wvfm")

