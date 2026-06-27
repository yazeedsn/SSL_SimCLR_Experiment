from torch import nn, Tensor
from torchvision.models import resnet18

class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x).flatten(1)
        return self.projection_head(x)
