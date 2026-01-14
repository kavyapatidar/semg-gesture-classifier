import os
import numpy as np

from src.data_loader import load_emg_file
from src.preprocessing import preprocess_emg
from src.windowing import create_windows


def extract_label(filename):
    return int(filename.split("_")[0].replace("gesture", ""))


def build_dataset_by_subject(data_root, fs=512, window_ms=200, overlap=0.5):
    """
    Build dataset grouped by subject.
    Returns:
        subject_data[subject_id] = (X, y)
    """

    window_size = int((window_ms / 1000) * fs)
    step_size = int(window_size * (1 - overlap))

    subject_data = {}

    for session in os.listdir(data_root):
        session_path = os.path.join(data_root, session)
        if not os.path.isdir(session_path):
            continue

        for subject in os.listdir(session_path):
            subject_path = os.path.join(session_path, subject)
            if not os.path.isdir(subject_path):
                continue

            subject_id = subject  # keep full subject name

            if subject_id not in subject_data:
                subject_data[subject_id] = {"X": [], "y": []}

            for file in os.listdir(subject_path):
                if not file.endswith(".csv"):
                    continue

                label = extract_label(file)
                file_path = os.path.join(subject_path, file)

                data = load_emg_file(file_path)
                signal = preprocess_emg(data.values, fs).astype(np.float32)

                windows = create_windows(signal, window_size, step_size)

                subject_data[subject_id]["X"].append(windows)
                subject_data[subject_id]["y"].extend([label] * windows.shape[0])

    # Concatenate per subject
    for subject_id in subject_data:
        subject_data[subject_id]["X"] = np.concatenate(
            subject_data[subject_id]["X"], axis=0
        )
        subject_data[subject_id]["y"] = np.asarray(
            subject_data[subject_id]["y"], dtype=np.int64
        )

    return subject_data
