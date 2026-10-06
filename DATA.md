## 1. Official dataset URL

- **Source:** [moondream/glaucoma-detection](https://huggingface.co/datasets/moondream/glaucoma-detection) — Hugging Face Datasets Hub.
- **Modality:** retinal fundus images (RGB) + label (glaucoma severity class).
- **Format on the Hub:** Parquet files, loaded through the `datasets` library.

## 2. Dataset version

- Version used: the dataset revision available on the Hugging Face Hub, verified directly by loading it on **25-09-2026**.
- No separate versioned release/tag is published by the dataset authors; `datasets.load_dataset()` pins to the current main revision on the Hub unless a `revision=` commit hash is passed. The exact commit hash of the revision used for the experiments (retrieved via `huggingface_hub.HfApi().dataset_info("moondream/glaucoma-detection").sha`):
```
  56d14df317198383ba7ff7005e8a6cb904785a67
```

## 3. Data split

The dataset is pre-split by the authors into 3 subsets, used as-is (no re-splitting performed). Confirmed directly from `load_dataset("moondream/glaucoma-detection")` on 25-09-2026:

| Split | Images |
|---|---|
| Train | 2,847 |
| Validation | 1,259 |
| Test | 1,272 |
| **Total** | **5,378** |

- **Features:** `image` (JPEG bytes, `decode=False` — must be decoded manually, see Section 4), `class` (string).
- **Classes (3):** `"normal"`, `"early"`, `"advanced"` — glaucoma severity, mapped to integer indices `0, 1, 2` respectively (see `LABEL_MAP` in `src/datasets/glaucoma_dataset.py`). All 3 models must use this exact mapping so a given index means the same severity everywhere.
- **Per-class image counts:**
  | Split | normal | early | advanced |
  |---|---|---|---|
  | Train | 1,004 | 677 | 1,166 |
  | Validation | 403 | 312 | 544 |
  | Test | 574 | 262 | 436 |
  
  The dataset is moderately imbalanced — `early` is the minority class in every split (roughly half the size of `advanced`), which is consistent with `early`'s low recall being the hardest class to get right during model training.

## 4. Preprocessing procedure

Applied identically to all 3 models (CNN from scratch, DenseNet121, ResNet50), implemented in `src/datasets/glaucoma_dataset.py`:

1. **Decode the image.** The `image` column is stored with `decode=False`, i.e. each sample's `image` is a dict holding raw JPEG bytes (`{"bytes": ..., "path": None}`), not an auto-decoded image. It is opened manually with `PIL.Image.open(io.BytesIO(sample["image"]["bytes"])).convert("RGB")`.
2. **Map the label.** The `class` column holds strings (`"normal"`/`"early"`/`"advanced"`), not integers. Mapped to `0`/`1`/`2` via `LABEL_MAP` before being used in `nn.CrossEntropyLoss`.
3. **Deduplicate** near-identical/duplicate samples via `_drop_duplicates()` in `src/datasets/glaucoma_dataset.py`.
4. **Normalization** (all splits): per-channel mean/std from ImageNet statistics —
   `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.
5. Convert to PyTorch tensors (`transforms.ToTensor()`), batched via `torch.utils.data.DataLoader`.

Where CNN-from-scratch, DenseNet121 and ResNet50 differ:
 
- **CNN from scratch / DenseNet121:** resize to **224×224** via the shared `build_dataloaders()` pipeline, plus train-split-only augmentation — random horizontal flip, random rotation (±10°), color jitter (brightness ±0.1, contrast ±0.1, saturation ±0.05).
- **ResNet50 (the selected final model):** resize to **320×320** instead, via its own `eval_transform`/train transform defined directly in `notebooks/train_resnet50.ipynb` (not through `build_dataloaders()`) — this was the model picked for the best accuracy/robustness trade-off, so the XAI notebook (`xai_analysis.ipynb`) also evaluates at 320×320 to match exactly what the model was trained/evaluated on.

No manual relabeling or class rebalancing is applied — the dataset's original labels and splits are used directly (only exact/near-duplicate rows are dropped, per step 3 above).

## 5. Scripts to reproduce the data pipeline

Everything below is implemented in [`src/datasets/glaucoma_dataset.py`](src/datasets/glaucoma_dataset.py) and [`src/training/config.py`](src/training/config.py). To reproduce the exact train/val/test tensors used in the experiments:

```python
from src.datasets.glaucoma_dataset import build_dataloaders

train_loader, val_loader, test_loader = build_dataloaders()
```

This single call:
- Downloads the dataset from the Hugging Face Hub (`datasets.load_dataset("moondream/glaucoma-detection")`).
- Wraps each split in the `GlaucomaDataset` class (`src/datasets/glaucoma_dataset.py`).
- Applies the 224x224 shared transforms described in Section 4 above.
- Returns the 3 ready-to-use `DataLoader`s, with batch size set in `src/training/config.py` (`BATCH_SIZE`).

This is used as-is for the CNN-from-scratch and DenseNet121 pipelines. For **ResNet50** (the selected final model) and for `xai_analysys.ipynb`, the 320×320 pipeline is built directly in those notebooks instead — using the same `GlaucomaDataset`/`_drop_duplicates()` building blocks from `src/datasets/glaucoma_dataset.py`, but with a custom `eval_transform`/train transform at 320×320 rather than calling `build_dataloaders()`.

No new or separately processed dataset is published by this team — the original Hugging Face dataset is used directly through the pipeline above, so no additional downloadable link is provided beyond the official dataset URL in Section 1.

A handful of representative images are also committed under `demo_samples/` (not the full dataset) so the live presentation demo does not depend on re-downloading data or on an internet connection — see the README, section "Demo for the presentation".
