import pandas as pd

def load_emg_file(file_path):
    """
    Load EMG CSV file with proper headers and numeric dtype.
    Returns DataFrame of shape (samples, 8).
    """
    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)  # ← IMPORTANT: allow header
    elif file_path.endswith(".xlsx"):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

    # Force numeric conversion (safety)
    data = data.apply(pd.to_numeric, errors="coerce")

    # Drop rows with NaNs (if any)
    data = data.dropna()

    return data
