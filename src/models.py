from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

def get_baseline_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(in_features=512, out_features=10)
    return model

def get_simclr_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
    )
    return model