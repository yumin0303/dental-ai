import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DentalDataset(Dataset):
    def __init__(self, csv_file, min_confidence=0.0, transform=None):
        df = pd.read_csv(csv_file)
        if "confidence" in df.columns and min_confidence > 0:
            df = df[df["confidence"].astype(float) >= min_confidence].reset_index(drop=True)
        self.image_paths = df["image_path"].tolist()
        self.labels      = df["crowding"].astype(int).tolist()
        self.transform   = transform or BASE_TRANSFORM

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), torch.tensor(self.labels[idx], dtype=torch.float)
