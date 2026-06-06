import torch
from torch import nn
from torchvision import transforms as TF
from torchvision.transforms import v2


def get_to_tensor_transform():
    return v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ])

def get_agumnetation_normalized_transform():
    return TF.Compose([
        TF.RandomResizedCrop(96),
        TF.ColorJitter(),
        TF.ToTensor(),
        TF.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_random_agumentation_transform():
    return nn.Sequential(
        v2.RandomResizedCrop(96, scale=(0.3, 1)),
        v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
        v2.RandomApply([v2.GaussianBlur(3)]),
        v2.RandomHorizontalFlip(),
        v2.RandomGrayscale(0.2)
    )

def get_random_agumentation_normalized_transform():
    return nn.Sequential(
        get_random_agumentation_transform(),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    )


def get_normalization_transform():
    return TF.Compose([
        TF.ToTensor(),
        TF.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])