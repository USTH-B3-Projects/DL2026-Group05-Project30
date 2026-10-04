from torch.utils.data import Dataset, DataLoader    
from torchvision import transforms

from PIL import Image
from io import BytesIO

LABEL_MAP = {
    "normal": 0,
    "early": 1,
    "advanced": 2
}

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), 

    #Data augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degreees=10),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.05
    ),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class GlaucomaDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image_bytes = sample["image"]["bytes"]
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        label = LABEL_MAP[sample["class"]]

        if self.transform:
            image = self.transform(image)
        return image, label

def create_datasets(ds):
    train_data = GlaucomaDataset(ds["train"], transform=train_transform)
    val_data = GlaucomaDataset(ds["validation"], transform=eval_transform)
    test_data = GlaucomaDataset(ds["test"], transform=eval_transform)

    return train_data, val_data, test_data

def create_dataloaders(ds, batch_size=32):
    train_data, val_data, test_data = create_datasets(ds)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader    