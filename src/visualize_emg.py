import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_emg_file
from src.preprocessing import preprocess_emg

def plot_emg_before_after(file_path, sampling_rate=512, duration_ms=200):
    data = load_emg_file(file_path)
    raw = data.values

    num_samples = int((duration_ms / 1000) * sampling_rate)
    raw = raw[:num_samples, :]

    filtered = preprocess_emg(raw, sampling_rate)
    time = np.arange(num_samples) / sampling_rate

    plt.figure(figsize=(14, 12))

    for ch in range(raw.shape[1]):
        plt.subplot(raw.shape[1], 2, 2*ch + 1)
        plt.plot(time, raw[:, ch])
        plt.title(f"Ch{ch+1} Raw")
        plt.grid(True)

        plt.subplot(raw.shape[1], 2, 2*ch + 2)
        plt.plot(time, filtered[:, ch])
        plt.title(f"Ch{ch+1} Filtered")
        plt.grid(True)

    plt.suptitle("EMG Before vs After Filtering", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
