import os
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from src.emg_subject_dataset import EMGSubjectDataset
from src.cnn_model import EMGCNN


def subject_independent_cnn(data_root, epochs=6, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Collect unique subjects
    subjects = set()
    for session in os.listdir(data_root):
        session_path = os.path.join(data_root, session)
        if not os.path.isdir(session_path):
            continue
        for subject in os.listdir(session_path):
            subjects.add(subject)

    subjects = sorted(list(subjects))
    accuracies = []

    for test_subject in subjects:
        print(f"\nTesting on unseen subject: {test_subject}")

        train_subjects = [s for s in subjects if s != test_subject]
        test_subjects = [test_subject]

        train_dataset = EMGSubjectDataset(data_root, train_subjects)
        test_dataset = EMGSubjectDataset(data_root, test_subjects)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        model = EMGCNN(num_classes=5).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = torch.argmax(model(xb), dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        accuracies.append(acc)
        print(f"Accuracy for {test_subject}: {acc:.4f}")

        del model, train_loader, test_loader
        torch.cuda.empty_cache()

    print("\n==============================")
    print("Subject-Independent CNN Results")
    print("==============================")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
    print(f"Std Deviation: {np.std(accuracies):.4f}")

    return accuracies
