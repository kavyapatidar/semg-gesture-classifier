import os
import numpy as np

from src.data_loader import load_emg_file
from src.preprocessing import preprocess_emg
from src.windowing import create_windows


def extract_label(filename):
    """
    Extract gesture label from filename.
    Example: gesture00_trial01.csv -> 0
    """
    return int(filename.split("_")[0].replace("gesture", ""))


def build_dataset(data_root, fs=512, window_ms=200, overlap=0.5):
    """
    Build EMG dataset using windowing.
    Returns:
        X -> shape (N, window_samples, channels)
        y -> shape (N,)
    """

    window_size = int((window_ms / 1000) * fs)
    step_size = int(window_size * (1 - overlap))

    X_windows = []
    y_labels = []

    for session in os.listdir(data_root):
        session_path = os.path.join(data_root, session)
        if not os.path.isdir(session_path):
            continue

        for subject in os.listdir(session_path):
            subject_path = os.path.join(session_path, subject)
            if not os.path.isdir(subject_path):
                continue

            for file in os.listdir(subject_path):
                if not file.endswith(".csv"):
                    continue

                label = extract_label(file)
                file_path = os.path.join(subject_path, file)

                # Load + preprocess
                data = load_emg_file(file_path)
                signal = preprocess_emg(data.values, fs)

                # IMPORTANT: reduce memory
                signal = signal.astype(np.float32)

                windows = create_windows(signal, window_size, step_size)

                X_windows.append(windows)
                y_labels.extend([label] * windows.shape[0])

    # Concatenate safely
    X = np.concatenate(X_windows, axis=0)
    y = np.asarray(y_labels, dtype=np.int64)

    return X, y
