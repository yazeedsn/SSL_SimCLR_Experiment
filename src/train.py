import os 
from pathlib import Path
from collections.abc import Callable

import torch
from tqdm import tqdm
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def _run_epoch_linear_probe(
        model: Module,
        train_dl: DataLoader,
        optimizer: Optimizer,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        device: str,
        epoch: int,
        n_epochs: int,
) -> tuple[float, float]:
    total = 0
    running_loss = 0.0
    true_predictions = 0

    pbar = tqdm(train_dl)
    for batch_idx, (X, Y) in enumerate(pbar):
        X, Y = X.to(device), Y.to(device)
        Y_pred = model(X)
        loss = loss_fn(Y_pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += len(X)
        true_predictions += (Y == Y_pred.argmax(dim=1)).sum().item()
        running_loss += loss.item()

        pbar.set_description(f"Epoch [{epoch + 1}/{n_epochs}] Batch [{batch_idx + 1}/{len(train_dl)}]")
        pbar.set_postfix(Loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(train_dl)
    avg_acc = true_predictions / total
    return avg_loss, avg_acc


def _run_epoch_ssl(
        model: Module,
        train_dl: DataLoader,
        optimizer: Optimizer,
        augmentation: Callable[[Tensor], Tensor],
        loss_fn: Callable[..., Tensor],
        device: str,
        epoch: int,
        n_epochs: int,
        temperature: float,
) -> float:
    running_loss = 0.0

    pbar = tqdm(train_dl)
    for batch_idx, (X, _) in enumerate(pbar):
        X = X.to(device)
        X1 = augmentation(X)
        X2 = augmentation(X)
        z1 = model(X1)
        z2 = model(X2)

        loss = loss_fn(z1, z2, temperature=temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_description(
            f"Epoch [{epoch}/{n_epochs}] Loss: {running_loss / (batch_idx + 1):.4f}"
        )

    return running_loss / len(train_dl)


def train_linear_probe(
        model: Module,
        train_dl: DataLoader,
        optimizer_factory: Callable[[list[Parameter]], Optimizer],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        device: str,
        n_epochs: int,
) -> tuple[list[float], list[float]]:
    """Train only the final fully-connected layer(s) of `model` (linear probing).

    The optimizer should already reference only the
    unfrozen parameters, or be rebuilt after calling this function.

    Takes an optimizer factory to build an optimizer according to the unfreezed parameters, e.g. lambda p: Adam(p, lr=1e-3).

    Args:
        model:      A model whose linear head layers are prefixed with "fc".
        train_dl:   DataLoader yielding (inputs, labels) batches.
        optimizer_factory:  An optimizer builder function with the signature ``list[torch.nn.Parameter] -> torch.optim.Optimizer``.
        loss_fn:    Loss function with signature ``(predictions, targets) -> scalar``.
        device:     Device string, e.g. ``"cuda"`` or ``"cpu"``.
        n_epochs:   Number of training epochs.

    Returns:
        A ``(loss_history, accuracy_history)`` tuple, each a list of per-epoch averages.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_factory(trainable)
    
    model.train()

    loss_hist: list[float] = []
    acc_hist: list[float] = []

    for epoch in range(n_epochs):
        avg_loss, avg_acc = _run_epoch_linear_probe(
            model, train_dl, optimizer, loss_fn, device, epoch, n_epochs
        )
        loss_hist.append(avg_loss)
        acc_hist.append(avg_acc)
        print(f"\nEpoch {epoch + 1} completed — Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}\n")

    return loss_hist, acc_hist


def train_ssl(
        model: Module,
        train_dl: DataLoader,
        optimizer: Optimizer,
        augmentation: Callable[[Tensor], Tensor],
        loss_fn: Callable[..., Tensor],
        device: str,
        n_epochs: int,
        temperature: float = 0.1,
        resume_from_checkpoint: str | None = None,
        checkpoint_every: int | None = None
) -> list[float]:
    """Train a model with a self-supervised contrastive objective.

    Two augmented views of each batch are generated and passed through
    `model`; `loss_fn` receives the resulting embeddings and a temperature
    scalar.

    Args:
        model:         Encoder network to train.
        train_dl:      DataLoader yielding ``(inputs, _)`` batches.
        optimizer:     An optimizer configured for ``model``'s parameters.
        augmentation:  A stochastic transform applied to a tensor, returning a tensor.
        loss_fn:       Contrastive loss with signature
                       ``(z1, z2, temperature=...) -> scalar``.
        device:        Device string, e.g. ``"cuda"`` or ``"cpu"``.
        n_epochs:      Number of training epochs.
        temperature:   Temperature scalar forwarded to ``loss_fn``.
        resume_from_checkpoint:
            Path to a checkpoint file used to resume training. If provided,
            the model, optimizer, epoch, and loss history are restored from it.
        checkpoint_every:
            Save a checkpoint every `checkpoint_every` epochs. If `None`,
            intermediate checkpoints are not saved.

    Returns:
        A list of average losses, one per epoch.
    """
    loss_hist: list[float] = []
    epoch = 1
    if resume_from_checkpoint:
        cp = torch.load(resume_from_checkpoint, map_location=device)
        epoch = cp['epoch'] + 1
        model.load_state_dict(cp['model_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        loss_hist = cp['loss_history']
    
    checkpoints_dir =  Path('outputs/checkpoints')
    if checkpoint_every:
        os.makedirs(checkpoints_dir, exist_ok=True)
    model.train()

    for epoch in range(epoch, n_epochs + 1):
        avg_loss = _run_epoch_ssl(
            model, train_dl, optimizer, augmentation, loss_fn,
            device, epoch, n_epochs, temperature,
        )
        loss_hist.append(avg_loss)
        print(f"\nEpoch {epoch} completed — Avg Loss: {avg_loss:.4f}\n")
        if checkpoint_every is not None and epoch % checkpoint_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_history": loss_hist,
            },
                checkpoints_dir / f'sim_clr_cp_epoch_{epoch:03d}.pt'
            )
            print(f'Checkpoint saved for epoch {epoch}')

    return loss_hist