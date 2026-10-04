from torch.utils.data import Dataset, DataLoader    
from torchvision import transforms
from datasets import load_dataset

from PIL import Image
from io import BytesIO

import hashlib

LABEL_MAP = {
    "normal": 0,
    "early": 1,
    "advanced": 2
}
IDX_TO_CLASS = {v: k for k, v in LABEL_MAP.items()}

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), 

    # Data augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
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

def _hash_image(sample) -> str:
    return hashlib.md5(sample["image"]["bytes"]).hexdigest()

def _drop_duplicates(hf_split):
    seen = set()
    keep_indices = []
    for i, sample in enumerate(hf_split):
        h = _hash_image(sample)
        if h not in seen:
            seen.add(h)
            keep_indices.append(i)
    return hf_split.select(keep_indices)

def build_dataloaders(batch_size: int = 32):
    ds = load_dataset("moondream/glaucoma-detection")
    
    # Check for leakage across splits BEFORE deduplicating each split
    train_hashes = {_hash_image(s) for s in ds["train"]}
    val_hashes   = {_hash_image(s) for s in ds["validation"]}
    test_hashes  = {_hash_image(s) for s in ds["test"]}

    leak_train_val  = train_hashes & val_hashes
    leak_train_test = train_hashes & test_hashes
    if leak_train_val or leak_train_test:
        print(f"[WARNING] Detected duplicate images between train/val: {len(leak_train_val)}, "
              f"train/test: {len(leak_train_test)} -- needs to be processed before training.")

    train_split = _drop_duplicates(ds["train"])
    val_split   = _drop_duplicates(ds["validation"])
    test_split  = _drop_duplicates(ds["test"])

    print(f"After dedup: train {len(train_split)} (was {len(ds['train'])}), "
          f"val {len(val_split)} (was {len(ds['validation'])}), "
          f"test {len(test_split)} (was {len(ds['test'])})")

    # Build datasets from the DEDUPLICATED splits, not the raw ones
    train_ds = GlaucomaDataset(train_split, transform=train_transform)
    val_ds   = GlaucomaDataset(val_split, transform=eval_transform)
    test_ds  = GlaucomaDataset(test_split, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader    
