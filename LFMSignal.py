import numpy as np
from scipy.signal import chirp, spectrogram
import matplotlib.pyplot as plt
import logging

log = logging.getLogger(__name__)


class Waveform(object):
    r"""
    An abstract class representing a :class:`Waveform`.
    """
    _repr_indent = 4

    def __init__(self, fs, data=None):
        self.fs = fs
        self.data = data

    #def __getitem__(self, index):
    #    raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        head = "Waveform " + self.__class__.__name__
        body = ["Length: {} samples".format(self.__len__())]
        body += ["Sampling Rate: {} Hz".format(self.fs)]

        if hasattr(self, "seconds") and self.seconds is not None:
            body += ["Time: {} seconds".format(self.seconds)]

        if hasattr(self, "voltage") and self.voltage is not None:
            body += ["Voltage: {} V".format(repr(self.voltage))]

        if hasattr(self, "power") and self.power is not None:
            body += ["Power {} watts".format(repr(self.power))]

        if hasattr(self, "powerdb") and self.powerdb is not None:
            body += ["{} dB".format(repr(self.powerdb))]

        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

class LFM_Waveform(Waveform):
    """
    Base LFM class from Waveform

    """
    def __init__(self, fs, f0, f1, seconds, voltage=1):
        super().__init__(fs)
        self.fs = fs
        self.f0 = f0
        self.f1 = f1
        self.voltage = voltage
        self.seconds = seconds
        t = np.linspace(0, seconds, seconds * fs, endpoint=False)
        self.data = chirp(t, f0=f0, f1=f1, t1=seconds, method='linear')
        self.data *= voltage
        #self.x_watts = self.data ** 2
        self.power = np.mean(self.data ** 2)
        self.powerdb = 10 * np.log10(self.power)

    def add_AWGN(self, target_snr_db=20):
        noise_avg_db = self.powerdb - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        mean_noise = 0
        self.noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(self.data))
        # Noise up the original signal
        self.data += self.noise

    def plot_wvfm(self):
        t = np.linspace(0, self.seconds, self.seconds * self.fs, endpoint=False)
        plt.plot(t, self.data)
        plt.xlabel('t (sec)')
        plt.show()

    def spectrogram(self, fs=None):
        if not fs:
            fs = self.fs
            print("here")
        f, t, Sxx = spectrogram(self.data, fs)
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Waveform sampled at {} Hz'.format(fs))
        plt.show()


def energy_spectral_density(signal, title='Holder'):
    # Normalized energy spectrum
    half_len = len(signal) / 2
    dsp1 = np.fft.fftshift(np.abs(np.fft.fft(signal)) ** 2)
    plt.plot(np.arange(-half_len, half_len, dtype=float) / float(half_len), dsp1)
    plt.xlim(0, 1)
    plt.title('{} Spectrum'.format(title))
    plt.ylabel('Squared modulus')
    plt.xlabel('Normalized Frequency')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    log = logging.getLogger('__main__')
    # logLevel = logging.DEBUG
    # logging.basicConfig(filename='l.log',
    #                     filemode='w',
    #                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    #                     datefmt='%H:%M:%S',
    #                     level=logLevel)

    # Signal properties
    fc_range = [1000, 2000]
    bw_range = [200, 500]

    # Nyquist Sampling for all signals
    nyquist_scale = 1
    fs = (max(fc_range) + max(bw_range)) * 2 * nyquist_scale
    print("Sampling at: {} Hz".format(fs))
    # Sampling
    fc = np.random.randint(fc_range[0], fc_range[1], size=[1])
    bw = np.random.randint(bw_range[0], bw_range[1], size=[1]) // 2
    f0 = fc - bw
    f1 = fc + bw
    voltage = 100
    seconds = 1

    lfm = LFM_Waveform(fs, f0, f1, seconds, voltage)
    log.debug(str(lfm))

    lfm.plot_wvfm()
    lfm.spectrogram()
    energy_spectral_density(lfm.data, "wvfm")

    # Add noise at SNR= -10db
    lfm.add_AWGN(-10)
    lfm.plot_wvfm()
    lfm.spectrogram()
    energy_spectral_density(lfm.data, "wvfm")

