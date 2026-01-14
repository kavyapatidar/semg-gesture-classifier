import os
import torch
import numpy as np

from torch.utils.data import Dataset
from src.data_loader import load_emg_file
from src.preprocessing import preprocess_emg
from src.windowing import create_windows


class EMGWindowDataset(Dataset):
    def __init__(self, data_root, fs=512, window_ms=200, overlap=0.5):
        self.samples = []

        window_size = int((window_ms / 1000) * fs)
        step_size = int(window_size * (1 - overlap))

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

                    label = int(file.split("_")[0].replace("gesture", ""))
                    file_path = os.path.join(subject_path, file)

                    data = load_emg_file(file_path)
                    signal = preprocess_emg(data.values, fs).astype(np.float32)

                    windows = create_windows(signal, window_size, step_size)

                    for w in windows:
                        self.samples.append((w, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = torch.tensor(x, dtype=torch.float32).permute(1, 0)  # (8, 102)
        y = torch.tensor(y, dtype=torch.long)
        return x, y
