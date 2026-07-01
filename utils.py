import torch 
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"

def _save_figure(fig: Figure, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_metric(
        yvalues: list[float],
        xvalues: list[float] | range | None = None,
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