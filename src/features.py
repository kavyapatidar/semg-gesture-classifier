import numpy as np


def extract_emg_features(window):
    """
    Extract standard EMG features from one window.
    window shape: (samples, channels)
    returns: 1D feature vector
    """

    features = []

    for ch in range(window.shape[1]):
        signal = window[:, ch]

        # Mean Absolute Value (MAV)
        mav = np.mean(np.abs(signal))

        # Root Mean Square (RMS)
        rms = np.sqrt(np.mean(signal ** 2))

        # Variance
        var = np.var(signal)

        # Waveform Length (WL)
        wl = np.sum(np.abs(np.diff(signal)))

        # Zero Crossing (ZC)
        zc = np.sum(np.diff(np.sign(signal)) != 0)

        features.extend([mav, rms, var, wl, zc])

    return np.array(features, dtype=np.float32)


def extract_features_from_windows(X):
    """
    X shape: (N, samples, channels)
    returns: (N, num_features)
    """

    feature_list = [extract_emg_features(window) for window in X]
    return np.vstack(feature_list)
