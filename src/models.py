from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

def get_baseline_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(in_features=512, out_features=10)
    return model

def get_simclr_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    projection_head = nn.Sequential(
        nn.Linear(in_features=512, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=128)
    )
    model.fc = projection_head
    return model