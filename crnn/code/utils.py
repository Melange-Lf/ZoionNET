import os
import pathlib
import torch

from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, List, Dict
import pandas as pd
import torchvision
import numpy as np

"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


import matplotlib.pyplot as plt

def plot_training_history(history):
    
    train_loss = history['train_loss']
    train_metric = history['train_metric'] 
    test_loss = history['test_loss']
    test_metric = history['test_metric']
    
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, test_loss, label='Test Loss', marker='o')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metric, label='Train Metric', marker='o')
    plt.plot(epochs, test_metric, label='Test Metric', marker='o')
    plt.title('Metric over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)



def loader_from_numpy(X,
                      y,
                      batch,
                      workers=16,
                      seed=42,
                      shuffle=False):
    
    torch.manual_seed(seed)
    Xt = torch.tensor(X).type(torch.float32)
    yt = torch.tensor(y).type(torch.float32)

    # if yt.dim() <2: 
    #     yt.unsqueeze(dim=-1)

    dataset = TensorDataset(Xt, yt)
    loader = DataLoader(dataset=dataset,
                           batch_size=batch,
                           num_workers=workers,
                           shuffle=shuffle)
    return loader





# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)
        



class NumpyFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.npy")) # note: you'd have to update this if you've got .png's or .jpeg's
        
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_numpy_array(self, index: int) -> Image.Image:
        "Opens an array via a path and returns it."
        array_path = self.paths[index]
        return np.load(array_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        array = self.load_numpy_array(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        
        return array, class_idx # return data, label (X, y)



# test_transforms = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor()
# ])

# test_data_custom = ImageFolderCustom(targ_dir=test_dir, 
#                                      transform=test_transforms)

# 1. Subclassesing
class SoftLabelImageCustom(Dataset):
    
    # 2. Initialize with a targ_dir (contains all images)
    def __init__(self, csv: pd.DataFrame, targ_dir: str, classes: int, extension='jpg') -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob(f"*.{extension}")) # note: you'd have to update this if you've got .png's or .jpeg's
        self.classes = classes
        self.csv = csv

        # Create a mapping from image identifier (here, file name) to soft labels.
        # Adjust if your CSV uses full paths or a different identifier.
        self.label_dict = {}
        for _, row in self.csv.iterrows():
            image_id = row["Path"]  # e.g. 'image1.jpg'
            # Assume that the next columns correspond to the soft-label probabilities.
            # This extracts the first 'classes' probability values.
            soft_label = row.iloc[1:1+classes].values.astype(float)
            self.label_dict[image_id] = soft_label

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        img_path  = self.paths[index].name

        soft_label = self.label_dict.get(img_path)
        soft_label_tensor = torch.tensor(soft_label, dtype=torch.float32)

        return torchvision.transforms.ToTensor(img), soft_label_tensor
    



    
# 1. Subclassesing
class SoftLabelFbankCustom(Dataset):
    
    # 2. Initialize with a targ_dir (contains all fbanks)
    def __init__(self, csv,  classes, target_dir = None) -> None:
        
        # 3. Create class attributes
        self.classes = classes
        if target_dir:
            self.target_dir = target_dir + "/"
        else:
            self.target_dir = ''
        
        csv = pd.read_csv(csv)
        self.csv = csv
        
        if len(self.csv.columns) > classes + 1:
            self.csv = self.csv.drop(columns=['true_label', 'predicted_label', 'correct'])

        # Create a mapping from fbank identifier (here, file name) to soft labels.
        # Adjust if your CSV uses full paths or a different identifier.
        self.label_dict = {}
        for _, row in self.csv.iterrows():
            fbank_id = row["file_path"]  # e.g. 'fbank1.pt'
            # Assume that the next columns correspond to the soft-label probabilities.
            # This extracts the first 'classes' probability values.
            soft_label = row.iloc[1:1+classes].values.astype(float)
            self.label_dict[fbank_id] = soft_label

    
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.label_dict.keys())
    
    def __getitem__(self, index: int):
        "Returns one sample of data, data and label (X, y)."
        fbank_path = self.csv.iloc[index]["file_path"]
        # print(self.target_dir, fbank_path)
        fbank = torch.load(self.target_dir + fbank_path).unsqueeze(2) # crnn expects a channel dimension, fbanks don't have that


        soft_label = self.label_dict.get(fbank_path)
        soft_label_tensor = torch.tensor(soft_label, dtype=torch.float32)

        return fbank, soft_label_tensor