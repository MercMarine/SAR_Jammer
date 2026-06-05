"""

Author - MercMarine
GitHub - https://github.com/MercMarine

signals.py - Классы для генерации сигналов помех с маской РСА

"""

import numpy as np
import matplotlib.pyplot as plt

class BaseSignal:

    # Базовый класс сигнала

    def __init__(self, amplitude=1.0, length=500, sample_rate=300,
                 pulse_duration=None, T_p=None, use_sar_mask=True, tau_sar=None):

        self.amplitude = amplitude
        self.length = length
        self.sample_rate = sample_rate
        self.time = np.arange(length) / sample_rate

        self.pulse_duration = pulse_duration if pulse_duration is not None else (length / self.sample_rate)
        self.T_p = T_p if T_p is not None else (length / self.sample_rate)
        self.use_sar_mask = use_sar_mask
        self.tau_sar = tau_sar if tau_sar is not None else self.pulse_duration

        self.tau_sar = min(self.tau_sar, self.T_p)

        self.signal = None

    def generate(self):
        raise NotImplementedError("Подклассы должны реализовать generate()")

    def apply_pulse_envelope(self):
        if self.signal is not None:
            envelope = (self.time <= self.pulse_duration).astype(float)
            self.signal *= envelope

    def apply_sar_mask(self):
        if self.signal is not None and self.use_sar_mask and self.T_p > 0:
            t_mod = np.mod(self.time, self.T_p)
            sar_mask = (t_mod <= self.tau_sar).astype(float)
            self.signal *= sar_mask

    def get_signal(self):
        if self.signal is None:
            self.generate()
        if self.use_sar_mask:
            self.apply_sar_mask()
        else:
            self.apply_pulse_envelope()

        return self.signal

class SinusoidalSignal(BaseSignal):
    def __init__(self, frequency=50, **kwargs):
        super().__init__(**kwargs)
        self.frequency = frequency

    def generate(self):
        self.signal = self.amplitude * np.exp(1j * 2 * np.pi * self.frequency * self.time)
        return self.signal

class LCHMSignal(BaseSignal):
    def __init__(self, f_start=10, f_end=90, f_center=None, bandwidth=None, **kwargs):
        super().__init__(**kwargs)
        if f_center is not None and bandwidth is not None:
            self.f_center = f_center
            self.bandwidth = bandwidth
            self.f_start = f_center - bandwidth / 2
            self.f_end = f_center + bandwidth / 2
        else:
            self.f_start = f_start
            self.f_end = f_end
            self.f_center = (f_start + f_end) / 2
            self.bandwidth = f_end - f_start

        self.chirp_rate = (self.f_end - self.f_start) / self.time[-1] if self.time[-1] != 0 else 0

    def generate(self):
        phase = 2 * np.pi * (self.f_start * self.time + 0.5 * self.chirp_rate * self.time ** 2)
        self.signal = self.amplitude * np.exp(1j * phase)
        return self.signal