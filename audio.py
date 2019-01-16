import librosa
import numpy as np


class AcousticExtractor:
    def __init__(self, fs=16000, fftl=512, shiftms=5.0,
                 win_len=25.0, minf0=40.0, maxf0=500.0,
                 n_mels=80):
        self.fs = fs
        self.fftl = fftl
        self.win_shift = int(shiftms*self.fs*1e-3)
        self.win_len = int(win_len*self.fs*1e-3)
        self.minf0 = minf0
        self.maxf0 = maxf0
        self.n_mels = n_mels

    def spectrogram(self, x, time_first=True):
        """
        :param x: np.ndarray[], real-valued waveform
        :param time_first: boolean. optional.if True,
                time axis is followed by bin axis.
                In this case, shape of returns is (t, 1 + n_fft/2)
        :return:
            mag: np.ndarray [shape=(t, 1 + n_fft/2) or (1 + n_fft/2, t)]
        Magnitude spectrogram.
            phase: np.ndarray [shape=(t, 1 + n_fft/2) or (1 + n_fft/2, t)]
        Phase spectrogram.
        """
        stft = librosa.core.stft(y=x,
                                 n_fft=self.fftl,
                                 hop_length=self.win_shift,
                                 win_length=self.win_len)
        mag = np.abs(stft)
        phase = np.angle(stft)

        if time_first:
            mag = mag.T
            phase = phase.T

        return mag, phase

    def spec2wav(self, mag, num_iters=50, phase=None, time_first=True):
        """
            Get a waveform from the magnitude spectrogram by Griffin-Lim Algorithm.
            Parameters
            ----------
            mag : np.ndarray [shape=(1 + n_fft/2, t)]
                Magnitude spectrogram.
            num_iters: int > 0 [scalar]
                Number of iterations of Griffin-Lim Algorithm.
            phase : np.ndarray [shape=(1 + n_fft/2, t)]
                Initial phase spectrogram.
            time_first: if mag is time first
            Returns
            -------
            wav : np.ndarray [shape=(n,)]
                The real-valued waveform.
        """
        assert num_iters > 0
        if phase is None:
            phase = np.pi * np.random.rand(*mag.shape)
        stft = mag * np.exp(1.j * phase)
        wav = None
        for i in range(num_iters):
            if time_first:
                stft_temp = stft.T
            else:
                stft_temp = stft
            wav = librosa.istft(stft_temp, win_length=self.win_len,
                                hop_length=self.win_shift)
            if i != num_iters - 1:
                stft = librosa.stft(wav, n_fft=self.fftl,
                                    win_length=self.win_len,
                                    hop_length=self.win_shift)
                _, phase = librosa.magphase(stft)
                phase = np.angle(phase)
                if time_first:
                    stft = mag * np.exp(1.j * phase).T
                else:
                    stft = mag * np.exp(1.j * phase)
        return wav

    def _spec2melspec(self, spec):
        """
           Convert a linear-spectrogram to mel-spectrogram.
           :param spec: Linear-spectrogram.
           :return: Mel-spectrogram.
        """
        mel_basis = librosa.filters.mel(self.fs, self.fftl, self.n_mels)
        mel = np.dot(mel_basis, spec)
        return mel

    def melspectrogram(self, x, time_first=True):
        """
        :param x: np.ndarray[], real-valued waveform
        :param time_first: boolean. optional.if True,
                time axis is followed by bin axis.
                In this case, shape of returns is (t, 1 + n_fft/2)
        :return: mel-spectrogram
        """
        mag_spec, phase_spech = self.spectrogram(x)
        mel_spec = self._spec2melspec(mag_spec)
        if time_first:
            mel_spec = mel_spec.T
        return mel_spec

    def npow(self, S=None, y=None, time_first=True):
        if S is not None:
            if time_first:
                S_ = S.T
            else:
                S_ = S
            pow = librosa.feature.rmse(S=S_, frame_length=self.win_len,
                                       hop_length=self.win_shift)
        elif y is not None:
            pow = librosa.feature.rmse(y=y, frame_length=self.win_len,
                                       hop_length=self.win_shift)
        else:
            raise ValueError("Both mag and y are None!!")

        return librosa.core.power_to_db(np.squeeze(pow), ref=np.max)
