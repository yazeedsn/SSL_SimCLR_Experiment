from torchvision.datasets import STL10
import transforms

def get_labeled_ds(data_path):
    return STL10(
        data_path, 
        split="train", 
        download=True, 
        transform=transforms.get_agumnetation_normalized_transform()
    )

def get_unlabeled_ds(data_path):
    return STL10(
        data_path, 
        split="unlabeled", 
        download=True, 
        transform=transforms.get_to_tensor_transform()
        )

def get_eval_ds(data_path):
    return STL10(
        data_path, 
        split="test", 
        download=True, 
        transform=transforms.get_normalization_transform()
    )