import torch
from torch.nn import Module
from torch.utils.data import DataLoader

def evaluate_linear_probe(
        model: Module,
        test_dl: DataLoader,
        test_ds_size: int,
        device: str,
) -> float:
    model.eval()
    true_predictions = 0

    with torch.no_grad():
        for X, Y in test_dl:
            X, Y = X.to(device), Y.to(device)
            Y_pred = model(X)
            true_predictions += (Y == Y_pred.argmax(dim=1)).sum().item()

    acc = true_predictions / test_ds_size
    print(f"Test Accuracy: {acc:.4f}")
    return acc