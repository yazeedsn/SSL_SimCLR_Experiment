import torch 
import math
from pathlib import Path
import matplotlib.pyplot as plt

# Plotting Utilities
def plot_metrics(
    metrics,
    epochs=None,
    figsize=(12, 5),
    save_path=None,
    show=True,
):
    """
    metrics:
        List of dictionaries with keys:
        {
            "values": ...,
            "title": ...,
            "ylabel": ...
        }
    """

    n_metrics = len(metrics)
    if epochs is None:
        epochs = range(1, len(metrics[0]["values"]) + 1)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.plot(epochs, metric["values"], marker="o")
        ax.set_title(metric["title"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric["ylabel"])
        ax.grid(True)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show: plt.show() 
    else: plt.close(fig)


def plot_image_grid(
    images,
    labels=None,
    n=10,
    cols=5,
    transform=None,
    figsize=None,
):
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
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
        axes[i].imshow(img)
        if labels is not None:
            axes[i].set_title(str(labels[i]))

    fig.tight_layout()
    plt.show()

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