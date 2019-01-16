import librosa
import numpy as np
from audio import AcousticExtractor
import matplotlib.pyplot as plt


if __name__ == '__main__':
    wav_f = 'C:\\Users\\LUHUI\\Desktop\\HCSI\\paper\\voice_conversion\\cmu_arctic\\cmu_us_bdl_arctic\\wav\\arctic_a0005.wav'
    wav_arr, sr = librosa.load(wav_f, sr=None)
    ae = AcousticExtractor()
    spec_mag, phase = ae.spectrogram(wav_arr)
    print(spec_mag.shape)
    s_pow = ae.npow(S=spec_mag)
    print(s_pow.shape)
    y_pow = ae.npow(y=wav_arr)
    print(y_pow.shape)
    plt.subplot('311')
    plt.plot(wav_arr)
    plt.subplot('312')
    plt.plot(s_pow)
    plt.subplot('313')
    plt.plot(y_pow)
    plt.show()
