import torch 
import math

import matplotlib.pyplot as plt
from torch import Tensor
from pathlib import Path

def _tensor_to_numpy(img: Tensor):
    img = img.detach().cpu()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)
    return img.numpy()

def _save_figure(fig: plt.Figure, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_metric(
        yvalues: list[float],
        xvalues: list[float] | None,
        title: str | None,
        ylabel: str | None,
        xlabel: str | None,
        figsize: tuple[int, int] = (6, 4),
        save_path: Path | None = None,
        show: bool = True,
) -> None:
    if not xvalues:
        xvalues = range(1, len(yvalues) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xvalues, yvalues, marker="o")
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(True)
    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, Path(save_path))
        
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_image_grid(
        images,
        labels: list | None = None,
        n: int = 10,
        cols: int = 5,
        transform=None,
        figsize: tuple[int, int] | None = None,
        save_path: Path | None = None,
        show: bool = True,
    ) -> None:
    n = min(n, len(images))
    rows = math.ceil(n / cols)
    if figsize is None:
        figsize = (cols * 2.5, rows * 2.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]
    for ax in axes:
        ax.axis("off")

    for i in range(n):
        img = images[i]
        if transform is not None:
            img = transform(img)
        if isinstance(img, Tensor):
            img = _tensor_to_numpy(img)
        axes[i].imshow(img)
        if labels is not None:
            axes[i].set_title(str(labels[i]))

    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, Path(save_path))

    if show:
        plt.show()
    else:
        plt.close(fig)

# Saving/Loading Utilities 
def save_checkpoint(model, optimizer, epoch, loss, path="outputs/checkpoints/checkpoint.pt"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path="outputs/checkpoints/checkpoint.pt", device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, epoch, loss