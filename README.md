# Explainable Glaucoma Stage Classification

Classify glaucoma severity from retinal fundus images, then apply XAI (Explainable AI) methods to visualize and explain which regions drive the model's predictions.
[Report](https://www.overleaf.com/7782515589vfbbncshxcbh#3e6d5a)

## 1. Project overview

- **Dataset:** [moondream/glaucoma-detection](https://huggingface.co/datasets/moondream/glaucoma-detection) (Hugging Face) — glaucoma severity classification, 5,380 fundus images, 416×416 resolution, pre-split into train/val/test (2,850 / 1,260 / 1,270).
- **Pipeline:**
  1. Train 3 models: **CNN built from scratch**, **DenseNet121**, **ResNet50** (the latter two via transfer learning).
  2. Compare their performance and select the **best-performing model**.
  3. Apply **3 XAI methods** (still being decided, will be updated) to the best model to explain its predictions.
- **Main library:** PyTorch (`torch`, `torchvision`).
- **Training environment:** Google Colab / Kaggle Notebooks / Jupyter Notebook — code is still written and version-controlled in this repo (edited in VSCode, actually run on the notebook platforms).

## 2. Team assignments

| Member | Model | Notes |
|---|---|---|
| Quang & Mai Anh | ResNet50 | Transfer learning, fine-tuning |
| Nhat & Viet | DenseNet121 | Transfer learning, fine-tuning |
| An & Ngan Anh | CNN from scratch | Custom architecture |

> Once the 3 models are trained and compared, the team will meet again to finalize the **3 XAI methods** and assign pairs to each one (2 people per method). This section will be updated once that's decided.

## 3. Repository structure

```
.
├── data/                       # Raw data is NOT committed to git (see .gitignore)
│   ├── raw/                        # Original dataset downloaded manually from Hugging Face
│   └── processed/                  # Preprocessed data (if any)
│   └── demo_samples/           # Sample images for live demos - IS committed to git
│
├── notebooks/                  # Notebooks run on Colab/Kaggle/Jupyter, kept for reference/reproducibility
│   ├── data_exploration.ipynb
│   ├── train_cnn_scratch.ipynb
│   ├── train_densenet121.ipynb
│   ├── train_resnet50.ipynb
│   └── xai_analysis.ipynb
│
├── src/                        # Main source code, imported into notebooks
│   ├── datasets/
│   │   └── glaucoma_dataset.py     # Custom Dataset + DataLoader
│   ├── models/
│   │   ├── cnn_scratch.py          # Custom CNN architecture
│   │   ├── densenet121.py          # torchvision DenseNet121 wrapper + fine-tuning
│   │   └── resnet50.py             # torchvision ResNet50 wrapper + fine-tuning
│   ├── xai/
│   │   ├── method1.py              # Rename once the XAI method is decided (e.g. grad_cam.py)
│   │   ├── method2.py
│   │   └── method3.py
│   ├── training/
│   │   ├── train_loop.py           # Shared train_loop / test_loop functions
│   │   └── config.py               # Hyperparameters, paths, seed
│   └── utils/
│       ├── metrics.py              # Accuracy, F1, confusion matrix, etc.
│       └── visualize.py            # Plot heatmaps, compare original vs. explained images
│
├── checkpoints/                 # Trained model weights (.pt) — not committed to git
│
├── results/                     # Results: model comparison tables, XAI heatmaps, training logs
│   ├── metrics/
│   └── figures/
│
├── reports/                     # Report / presentation slides (LaTeX/PDF/pptx)
│
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

## 4. What each file in `src/` does
 
| File | Role |
|---|---|
| `src/training/config.py` | Shared paths, hyperparameters (batch size, learning rate, epochs, image size) and random seed. Everyone imports from here instead of hardcoding these values, so all 3 models are trained under the same settings. |
| `src/datasets/glaucoma_dataset.py` | Downloads the dataset from Hugging Face, defines image transforms (resize, normalize, augmentation), and exposes `build_dataloaders()` — returns train/val/test `DataLoader`s. Used identically by all 3 pairs so the data pipeline is never a source of unfair comparison. |
| `src/models/cnn_scratch.py`, `densenet121.py`, `resnet50.py` | Each defines **one model's architecture** — the thing each pair actually edits. `cnn_scratch.py` is a custom architecture built from `nn.Conv2d`/`nn.Linear`; the other two wrap pretrained `torchvision.models` and replace the final layer for fine-tuning. |
| `src/training/train_loop.py` | **Does not define or create a model** — it receives an already-built model as an argument and runs the training process on it (`train_one_epoch`, `evaluate`, `fit`). The same training procedure (same order of `zero_grad`/`backward`/`step`, same loss handling) is reused for all 3 models, called separately in each pair's own notebook. |
| `src/utils/metrics.py` | Runs **after** training/evaluation. Takes the predictions + true labels that `evaluate()` produced and turns them into accuracy, F1, and a confusion matrix, then saves them to `results/metrics/<model_name>.json` so the 3 models can be compared side by side. |
| `src/utils/visualize.py` | Plotting helpers: training-curve plots (train/val loss per epoch from `fit()`'s `history`), confusion-matrix heatmaps, and later, XAI heatmaps overlaid on the original image once the 3 XAI methods are chosen. |
| `src/xai/*.py` | Placeholder files, to be renamed/filled once the 3 XAI methods are finalized (e.g. `grad_cam.py`). |
 
**How they connect**, in the order a notebook actually calls them:
 
```
config.py             →  shared settings used by everything below
glaucoma_dataset.py   →  build_dataloaders() → train_loader, val_loader, test_loader
models/<model>.py     →  build_<model>()      → model
train_loop.py         →  fit(model, train_loader, val_loader, ...) → trains the model
                      →  evaluate(model, test_loader, ...)         → preds, labels
metrics.py            →  compute_metrics(labels, preds) → save_metrics(...)
visualize.py          →  plots from history / metrics
```

## 5. Libraries used

**Modeling / training:**
- `torch`, `torchvision` — model definitions, DataLoaders, training loop
- `torchvision.models` — load pretrained (ImageNet) DenseNet121 and ResNet50 for transfer learning
- `scikit-learn` — metrics (accuracy, precision/recall/F1, confusion matrix)
- `matplotlib`, `seaborn` — plots, result comparisons

**XAI (will be finalized once the 3 methods are chosen):**
- `pytorch-grad-cam` — Grad-CAM, Grad-CAM++, Eigen-CAM, Ablation-CAM (if CAM-based methods are chosen)
- `captum` — Integrated Gradients, Saliency, Occlusion (if Captum-based methods are chosen)

**Other:**
- `datasets` (Hugging Face) or `pandas` + `pillow` — load/read the dataset from Hugging Face (parquet files)
- `tqdm` — training progress bars

```
torch
torchvision
datasets
scikit-learn
matplotlib
seaborn
pandas
pillow
tqdm
pytorch-grad-cam
captum
```

> `requirements.txt` will be updated with the exact versions actually used for training.

## 6. Dataset documentation
 
See [`DATA.md`](DATA.md) for the official dataset URL, dataset version, data split, preprocessing procedure, and the exact script to reproduce the data used in the experiments — required for the course submission.

## 7. How to run

1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```
2. Create an environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset (run inside `notebooks/data_exploration.ipynb` or a dedicated script in `src/datasets/`):
   ```python
   from datasets import load_dataset
   ds = load_dataset("moondream/glaucoma-detection")
   ```
4. Train a model: open the corresponding notebook (`train_cnn_scratch.ipynb`, `train_densenet121.ipynb`, `train_resnet50.ipynb`) on Colab/Kaggle, import modules from `src/` (mount/clone the repo inside the notebook), and run the training loop.
5. Save checkpoints to `checkpoints/` and results/metrics to `results/`.
6. Once the best model is selected, run `05_xai_analysis.ipynb` to apply the 3 XAI methods and save the heatmaps to `results/figures/`.

## 8. Demo for the presentation
 
For the live demo in front of the lecturer, do **not** rely on `data/raw/` or on downloading the dataset on the spot — classroom Wi-Fi can be unreliable, and `data/raw/` is empty right after a fresh `git clone` (it's git-ignored).
 
Instead:
- Put 5–10 representative fundus images in `data/demo_samples/`. Unlike `data/raw/`, this folder **is** committed to git, so it's immediately available after cloning, on any machine.
- Point the demo script/notebook at `data/demo_samples/` instead of the full dataset when showing predictions + XAI heatmaps live.
- Test the demo once on a freshly cloned copy of the repo beforehand, to confirm it runs without needing the full dataset or an internet connection.

## 9. Collaboration conventions

- Everyone works on their own branch (`feature/resnet50`, `feature/densenet121`, `feature/cnn-scratch`, `feature/xai-<method>`, ...) and merges into `main` via Pull Request.
- Do not commit the raw dataset, model checkpoints (.pt/.pth), or other large files directly — add them to `.gitignore` and share via Drive/Kaggle Dataset if needed.
- Shared code (models, datasets, training loop, heatmap plotting functions) lives in `src/` to avoid copy-pasting between each person's notebook.
- Results comparing the 3 models (accuracy, F1, confusion matrix, etc.) are consolidated into a shared file in `results/metrics/` so the whole team can review it when choosing the best model.
