from torchvision.datasets import STL10
from .config import DATASET_PATH
from . import transforms

def get_labeled_ds():
    return STL10(
        DATASET_PATH, 
        split="train", 
        download=True, 
        transform=transforms.get_random_augmentation_normalized_transform()
    )

def get_unlabeled_ds():
    return STL10(
        DATASET_PATH, 
        split="unlabeled", 
        download=True, 
        transform=transforms.get_to_tensor_transform()
        )

def get_eval_ds():
    return STL10(
        DATASET_PATH, 
        split="test", 
        download=True, 
        transform=transforms.get_normalization_transform()
    )