#from cusignal import chirp, spectrogram
import numpy as np
# Bandwidt
from numpy import random
from LFMSignal import *
from scipy.signal import chirp, spectrogram, hilbert
import matplotlib.pyplot as plt

# What sampling rate should I use?

dataset = np.zeros((int(1e6), 2, 800), dtype=np.float32)
# Complex?
# generate 100,000, 10,000 each class 
#for label in range(10):

	# number of samples per class
	#for _ in range(10000): 
bws = np.linspace(4e6, 6e6-2e5, num=10)
label = random.randint(0,10)


seconds = pw = 50e-6
fs = 2 * 8e6
f0 = random.randint(0, 2e6)
f1 = f0 + bws[label]
t = np.linspace(0, seconds, seconds * fs, endpoint=False)
data = chirp(t, f0=f0, f1=f1, t1=seconds, method='linear')
data = hilbert(data).view('(2,)float')
#print(data.shape)
# plt.plot(t, data)
# plt.xlabel('t (sec)')
# plt.show()

exit(1)

# Signal properties
#fc_range = [10e9, 10e9]
#bw_range = [4e6, 6e6] # Signals are basebanded
#pw_range = [24e-6, 64e-6]
# Nyquist Sampling for all signals
nyquist_scale = 1
fs = (max(bw_range)) * 2 * nyquist_scale
print("Sampling at: {} Hz".format(fs))
print()
# Sampling
#fc = np.random.randint(fc_range[0], fc_range[1], size=[1])
#fc = 10e9e
#fs = 10e9
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
lfm.data = hilbert(lfm.data)
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

