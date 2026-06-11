import torch 
import math
from config import OUTPUT_DIR 

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
        xvalues: list[float] | None = None,
        title: str | None = None,
        ylabel: str | None = None,
        xlabel: str | None = None,
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
def save_model(model, name):
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f'{name}_weights.pth'
    torch.save(model.state_dict(), file_path)


def load_model(model, name, device="cpu"):
    path = Path(OUTPUT_DIR) / f'{name}_weights.pth'
    state_dict = torch.load(path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    return model