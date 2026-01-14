import numpy as np

def create_windows(signal, window_size, step_size):
    """
    Split EMG signal into overlapping windows.

    signal: numpy array (samples, channels)
    window_size: number of samples per window
    step_size: number of samples to slide
    """
    windows = []

    for start in range(0, signal.shape[0] - window_size + 1, step_size):
        window = signal[start:start + window_size, :]
        windows.append(window)

    return np.array(windows)
