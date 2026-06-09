# transforms.py
import torch
from torch import nn
from torchvision.transforms import v2

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

def _normalize() -> v2.Normalize:
    return v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

def get_to_tensor_transform() -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

def get_normalization_transform() -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        _normalize(),
    ])

def get_random_augmentation_transform() -> nn.Sequential:
    return nn.Sequential(
        v2.RandomResizedCrop(96, scale=(0.3, 1)),
        v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
        v2.RandomApply([v2.GaussianBlur(3)]),
        v2.RandomHorizontalFlip(),
        v2.RandomGrayscale(0.2),
    )

def get_random_augmentation_normalized_transform() -> nn.Sequential:
    return nn.Sequential(
        get_random_augmentation_transform(),
        _normalize(),
    )