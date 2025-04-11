import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob

class GestureImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Path to dataset folder containing labeled subfolders.
        transform: Image transformations.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {"left": 0, "right": 1, "stop": 2}  # Define label mapping

        # Iterate over subfolders
        for label_name in self.label_map.keys():
            folder_path = os.path.join(root_dir, label_name)  # e.g., "dataset/left"
            if os.path.isdir(folder_path):
                images = glob.glob(os.path.join(folder_path, "*.jpg"))  # Load images
                self.image_paths.extend(images)
                self.labels.extend([self.label_map[label_name]] * len(images))  # Assign label to each image
        
        print(f"Loaded {len(self.image_paths)} images from {len(self.label_map)} classes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)

        return image, label  # (Image Tensor, Label)

# Example Usage
# transform = transforms.Compose([
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),  # Converts to (C, H, W) where C=1 for grayscale
# ])

# dataset = GestureImageDataset(root_dir="dataset_green", transform=transform)
# print(len(dataset))
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# # Test the DataLoader
# for images, labels in dataloader:
#     print(images.shape)  # Expected: (batch_size, 1, 64, 64)
#     print(labels)  # Expected: (batch_size,)
#     print("hi")
#     break
