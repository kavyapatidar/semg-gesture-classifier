import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.emg_dataset import EMGWindowDataset
from src.cnn_model import EMGCNN


def train_cnn_model(data_root, epochs=10, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = EMGWindowDataset(data_root)

    # Train-test split indices
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx)
    )

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx)
    )

    model = EMGCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = torch.argmax(model(xb), dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total
    print("\nCNN Accuracy (random split):", acc)

    return model
