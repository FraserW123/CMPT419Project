import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob

class GestureSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sequences = []
        self.label_map = {"left": 0, "right": 1, "stop": 2}

        for gesture_folder in sorted(os.listdir(root_dir)):
            seq_path = os.path.join(root_dir, gesture_folder)
            if os.path.isdir(seq_path): 
                label_name = gesture_folder.split("_")[0]  
                if label_name in self.label_map:  
                    self.sequences.append((seq_path, self.label_map[label_name]))
        
        print(f"Loaded {len(self.sequences)} sequences: {self.sequences}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_folder, label_idx = self.sequences[idx]
        image_paths = sorted(glob.glob(os.path.join(seq_folder, "*.jpg")))  # Check case sensitivity
        
        print(f"Loading {len(image_paths)} images from {seq_folder}")

        if len(image_paths) == 0:
            print(f"Warning: No images found in {seq_folder}")

        images = []
        for img_path in image_paths:
            image = Image.open(img_path).convert("L")  
            if self.transform:
                image = self.transform(image)
            images.append(image)

        if len(images) == 0:
            return None, None  # Avoid errors in stacking

        images = torch.stack(images)  
        return images, label_idx  

# Example Usage
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

dataset = GestureSequenceDataset(root_dir="dataset_green", transform=transform)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for images, labels in dataloader:
    if images is not None:
        print(images.shape)
        print(labels)
    break
