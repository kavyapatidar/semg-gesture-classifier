import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def bandpass_filter(signal, fs, lowcut=20, highcut=200, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if high >= 1:
        high = 0.99  # safety clamp

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=0)

def notch_filter(signal, fs, freq=50, quality=30):
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, signal, axis=0)

def preprocess_emg(signal, fs=512):
    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    return signal
