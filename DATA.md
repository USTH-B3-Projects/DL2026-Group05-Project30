## 1. Official dataset URL

- **Source:** [moondream/glaucoma-detection](https://huggingface.co/datasets/moondream/glaucoma-detection) — Hugging Face Datasets Hub.
- **Modality:** retinal fundus images (RGB) + label (glaucoma severity class).
- **Format on the Hub:** Parquet files, loaded through the `datasets` library.

## 2. Dataset version

- Version used: the dataset revision available on the Hugging Face Hub as of **2026-10-03** (the date the team finalized this dataset choice).
- No separate versioned release/tag is published by the dataset authors; `datasets.load_dataset()` pins to the current main revision on the Hub unless a `revision=` commit hash is passed. The exact commit hash used for the experiments will be recorded here once training starts, by running:
  ```python
  from datasets import load_dataset
  ds = load_dataset("moondream/glaucoma-detection")
  print(ds["train"].info.download_checksums)  # or capture the dataset's commit hash via the `huggingface_hub` API
  ```

## 3. Data split

The dataset is pre-split by the authors into 3 subsets, used as-is (no re-splitting performed):

| Split | Images |
|---|---|
| Train | 2,850 |
| Validation | 1,260 |
| Test | 1,270 |
| **Total** | **5,380** |

Image resolution as distributed: 416×416 pixels.

## 4. Preprocessing procedure

Applied identically to all 3 models (CNN from scratch, DenseNet121, ResNet50), implemented in `src/datasets/glaucoma_dataset.py`:

1. **Resize** every image to 224×224 (matches the input size expected by ImageNet-pretrained DenseNet121/ResNet50; the from-scratch CNN uses the same size for a fair comparison).
2. **Train split only — augmentation:**
   - Random horizontal flip.
   - Random rotation (±10°).
3. **Normalization** (all splits): per-channel mean/std from ImageNet statistics —
   `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.
4. Convert to PyTorch tensors (`transforms.ToTensor()`), batched via `torch.utils.data.DataLoader`.

No manual relabeling, filtering, or class rebalancing is applied — the dataset's original labels and splits are used directly.

## 5. Scripts to reproduce the data pipeline

Everything below is implemented in [`src/datasets/glaucoma_dataset.py`](src/datasets/glaucoma_dataset.py) and [`src/training/config.py`](src/training/config.py). To reproduce the exact train/val/test tensors used in the experiments:

```python
from src.datasets.glaucoma_dataset import build_dataloaders

train_loader, val_loader, test_loader = build_dataloaders()
```

This single call:
- Downloads the dataset from the Hugging Face Hub (`datasets.load_dataset("moondream/glaucoma-detection")`).
- Wraps each split in the `GlaucomaDataset` class (`src/datasets/glaucoma_dataset.py`).
- Applies the transforms described in Section 4 above.
- Returns the 3 ready-to-use `DataLoader`s, with batch size set in `src/training/config.py` (`BATCH_SIZE`).

No new or separately processed dataset is published by this team — the original Hugging Face dataset is used directly through the pipeline above, so no additional downloadable link is provided beyond the official dataset URL in Section 1.

A handful of representative images are also committed under `data/demo_samples/` (not the full dataset) so the live presentation demo does not depend on re-downloading data or on an internet connection — see the README, section "Demo for the presentation".
