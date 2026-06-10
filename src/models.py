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
    
    def get_linear_prob(self, n_classes: int) -> nn.Module:
        return nn.Sequential(
            self.encoder,
            nn.Flatten(),
            nn.Linear(512, n_classes),
        )

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
