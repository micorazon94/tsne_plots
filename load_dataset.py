import os
import random
import shutil
from torchvision.datasets import Flowers102
import torchvision.transforms as transforms
import torch
from PIL import Image
from torch.utils.data import DataLoader



class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset class that loads a specified dataset split (train, val, or test) for the Flowers102 dataset,
    with the option to limit the number of classes or specify particular classes to load.

    Parameters:
    - dataset_name: str, name of the dataset (e.g., 'flowers102').
    - split: str, one of ['train', 'val', 'test'] to specify which dataset split to load.
    - max_classes: int, maximum number of classes to include in the dataset. If -1, include all classes.
    - chosen_classes: list or None, specific classes to load. If None, load all or a random subset based on max_classes.
    """
    def __init__(self, dataset_name, split='train', max_classes=-1, chosen_classes=None):
        self.dataset_name = dataset_name
        self.split = split
        self.max_classes = max_classes
        self.chosen_classes = chosen_classes
        # Create the dataset and obtain class-to-index mapping
        dataset, class_to_idx = self.create_dataset()
        self.dataset = dataset
        self.num_classes = len(class_to_idx)
        self.class_to_idx = class_to_idx
    
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Retrieve a sample and its label by index
        image, label = self.dataset[idx]
        return image, label

    def check_images_downloaded(self, root_dir):
        """
        Check if the dataset images are downloaded and organized in the expected directories.

        Parameters:
        - root_dir: str, the root directory where the dataset is expected to be located.

        Returns:
        - bool: True if images are downloaded and organized, False otherwise.
        """
        expected_dirs = ['train', 'val', 'test']
        for split in expected_dirs:
            split_dir = os.path.join(root_dir, split)
            if not os.path.exists(split_dir) or not os.listdir(split_dir):
                return False
        return True

    def create_dataset(self):
        """
        Create the dataset for the specified split and parameters.

        Returns:
        - dataset: List of tuples containing images and their labels.
        - class_to_idx: Dictionary mapping class labels to indices.
        """
        if self.dataset_name == 'flowers102':
            root_dir = './data/flowers'
            # Check if images are downloaded; if not, download and organize them
            if not self.check_images_downloaded(root_dir):
                self.download_flowers_images(root_dir)
            return self.create_flowers_dataset(root_dir)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}." +
                             "\nPlease choose 'flowers102' or create your own dataset function.")

    def download_flowers_images(self, root_dir='./data/flowers'):
        """
        Download and organize the Flowers102 dataset images into train, val, and test directories.

        Parameters:
        - root_dir: str, the root directory where the dataset should be stored.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset_splits = ['train', 'val', 'test']
        for split in dataset_splits:
            dataset = Flowers102(root='./data', split=split, transform=transform, download=True)
            split_dir = os.path.join(root_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            for idx in range(len(dataset)):
                img, label = dataset[idx]
                class_dir = os.path.join(split_dir, str(label))
                os.makedirs(class_dir, exist_ok=True)

                img_path = dataset._image_files[idx]
                img_name = os.path.basename(img_path)
                target_path = os.path.join(class_dir, img_name)

                shutil.copy(img_path, target_path)

    def create_flowers_dataset(self, root_dir):
        """
        Create and return a filtered dataset based on chosen classes or maximum classes.

        Parameters:
        - root_dir: str, the root directory where the dataset is organized.

        Returns:
        - filtered_dataset: List of tuples containing images and their remapped labels.
        - class_to_idx: Dictionary mapping chosen class names to indices.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        split_folder = os.path.join(root_dir, self.split)

        if self.chosen_classes is None:
            # Load all classes if no specific classes are chosen
            self.chosen_classes = [d for d in os.listdir(split_folder) if os.path.isdir(os.path.join(split_folder, d))]
            # If max_classes is set, limit the number of classes
            if self.max_classes != -1:
                self.chosen_classes = random.sample(self.chosen_classes, self.max_classes)

        # Create class_to_idx mapping and filter dataset
        class_to_idx = {cls: idx for idx, cls in enumerate(self.chosen_classes)}
        filtered_dataset = []

        for cls in self.chosen_classes:
            class_dir = os.path.join(split_folder, str(cls))
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                image = transform(Image.open(img_path).convert('RGB'))
                label = class_to_idx[cls]
                filtered_dataset.append((image, label))

        return filtered_dataset, class_to_idx


def create_dataloader(dataset, batch_size: int, num_workers: int, train: bool):
    if train:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


if __name__ == "__main__":
    dataset_name='flowers102'
    split='train'
    max_classes=10
    # chosen_classes=None
    chosen_classes = ['72', '61', '44', '3', '99', '42', '78', '68', '31', '70']
    dataset = CustomDataset(dataset_name, split, max_classes, chosen_classes)
    print(f"dataset size: {len(dataset)}")
    print(dataset.class_to_idx)
    print(dataset.num_classes)