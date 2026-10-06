# Explainable Retinal Image Classification (Glaucoma Detection)

Classify glaucoma severity from retinal fundus images, then apply XAI (Explainable AI) methods to visualize and explain which regions drive the model's predictions.

## 1. Project overview

- **Dataset:** [moondream/glaucoma-detection](https://huggingface.co/datasets/moondream/glaucoma-detection) (Hugging Face) — glaucoma severity classification (3 classes: `normal`, `early`, `advanced`), pre-split into train/val/test (2,847 / 1,259 / 1,272).
- **Pipeline:**
  1. Train 3 models: **CNN built from scratch**, **DenseNet121**, **ResNet50** (the latter two via transfer learning).
  2. Compare their performance and select the **best-performing model**.
  3. Apply **3 XAI methods — LayerCAM, GradCAM++, Occlusion** — to the checkpoint of the selected model, in a single notebook that runs all 3 methods together.
- **Main library:** PyTorch (`torch`, `torchvision`).
- **Training environment:** Google Colab / Kaggle Notebooks / Jupyter Notebook — code is still written and version-controlled in this repo (edited in VSCode, actually run on the notebook platforms).

## 2. Repository structure

```
.
├── demo_samples/               
│
├── notebooks/                 
│   ├── data_exploration.ipynb
│   ├── train_cnn_scratch.ipynb
│   ├── train_densenet121.ipynb
│   ├── train_resnet50.ipynb        
│   └── xai_analysis.ipynb           
│
├── src/                        
│   ├── datasets/
│   │   └── glaucoma_dataset.py     
│   ├── models/
│   │   ├── cnn_scratch.py          
│   │   ├── densenet121.py          
│   │   └── resnet50.py             
│   ├── xai/                        
│   │   │                          
│   │   ├── __init__.py
│   │   ├── common.py               
│   │   ├── cam_methods.py          
│   │   └── occlusion.py            
│   ├── training/
│   │   ├── train_loop.py          
│   │   └── config.py    
│   ├── utils/
│       ├── metrics.py          
│       └── visualize.py  
│
├── checkpoints/                    # Trained model weights (.pt) — not committed to git (see .gitignore);
│                                 
├── results/                        # Results: model comparison tables, XAI heatmaps, training logs
│   ├── metrics/                     
│   └── figures/
│   └── xai/                   
│                                   
├── reports/                     
│
├── requirements.txt               # Python dependencies
├── pyproject.toml
├── uv.lock                        # Locked dependency versions (for `uv`, used by some members)
├── .python-version                # Pinned Python version (3.14) — read by `uv` and pyenv-style tools
├── .gitignore
├── DATA.md
└── README.md
```

## 3. What each file in `src/` does

| File | Role |
|---|---|
| `src/training/config.py` | Shared paths, hyperparameters (batch size, learning rate, epochs, image size) and random seed. Everyone imports from here instead of hardcoding these values, so all 3 models are trained under the same settings. |
| `src/datasets/glaucoma_dataset.py` | Downloads the dataset from Hugging Face, defines image transforms (resize, normalize, augmentation), and exposes `build_dataloaders()` — returns train/val/test `DataLoader`s. Used identically by all 3 pairs so the data pipeline is never a source of unfair comparison. |
| `src/models/cnn_scratch.py`, `densenet121.py`, `resnet50.py` | Each defines **one model's architecture** — the thing each pair actually edits. `cnn_scratch.py` is a custom architecture built from `nn.Conv2d`/`nn.Linear`; the other two wrap pretrained `torchvision.models` and replace the final layer for fine-tuning. |
| `src/training/train_loop.py` | **Does not define or create a model** — it receives an already-built model as an argument and runs the training process on it (`train_one_epoch`, `evaluate`, `fit`). The same training procedure (same order of `zero_grad`/`backward`/`step`, same loss handling) is reused for all 3 models, called separately in each pair's own notebook. |
| `src/utils/metrics.py` | Runs **after** training/evaluation. Takes the predictions + true labels that `evaluate()` produced and turns them into accuracy, F1, and a confusion matrix, then saves them to `results/metrics/<model_name>.json` so the 3 models can be compared side by side. |
| `src/utils/visualize.py` | Plotting helpers: training-curve plots (train/val loss per epoch from `fit()`'s `history`) and confusion-matrix heatmaps. |
| `src/xai/common.py` | Shared helpers for the 3 XAI methods on the selected final model (ResNet50): `load_model(checkpoint_name)` rebuilds the model and loads trained weights, `denormalize()` converts a normalized tensor back to a displayable `[0, 1]` RGB image, and `pick_best_checkpoint()` can pick the best seed automatically from saved metrics (kept available, not required now that the team locked in the seed `2026` checkpoint directly). |
| `src/xai/cam_methods.py`, `occlusion.py` | Run **after** the best model (ResNet50) is selected. `cam_methods.py` exposes `run_layercam()` and `run_gradcam_plusplus()` (both via `pytorch-grad-cam`, need gradients); `occlusion.py` exposes `run_occlusion()` (via `captum`, gradient-free). All 3 take the same `(model, input_tensor, target_class, rgb_float)`-style arguments and return an overlay image, so `notebooks/xai_analysis.ipynb` can call all 3 side by side on the same sample. |

**How they connect**, in the order a notebook actually calls them:

```
config.py             →  shared settings used by everything below
glaucoma_dataset.py   →  build_dataloaders() → train_loader, val_loader, test_loader
models/<model>.py     →  build_<model>() → model
train_loop.py         →  fit(model, train_loader, val_loader, ...) → trains the model
                      →  evaluate(model, test_loader, ...) → preds, labels
metrics.py            →  compute_metrics(labels, preds) → save_metrics(...)
visualize.py          →  plots from history / metrics
xai/common.py         →  load_model(checkpoint_name) → model, target_layers
xai/cam_methods.py    →  run_layercam(...), run_gradcam_plusplus(...) → overlay image
xai/occlusion.py      →  run_occlusion(...) → overlay image
                      →  (only run on the selected ResNet50 checkpoint, after comparison;
                      →   all 3 called together from notebooks/xai_analysis.ipynb)
```

## 4. Libraries used

**Modeling / training:**
- `torch`, `torchvision` — model definitions, DataLoaders, training loop
- `torchvision.models` — load pretrained (ImageNet) DenseNet121 and ResNet50 for transfer learning
- `scikit-learn` — metrics (accuracy, precision/recall/F1, confusion matrix)
- `matplotlib`, `seaborn` — plots, result comparisons

**XAI:**
- `pytorch-grad-cam` (pip package name: `grad-cam`) — provides `GradCAMPlusPlus`, `LayerCAM`, used in `src/xai/cam_methods.py`.
- `captum` — provides `Occlusion`, used in `src/xai/occlusion.py`.

**Other:**
- `datasets` (Hugging Face) or `pandas` + `pillow` — load/read the dataset from Hugging Face (parquet files)
- `tqdm` — training progress bars
- `gdown` — downloads the checkpoint from its Google Drive share link inside `xai_analysis.ipynb` (checkpoints are gitignored, so they can't be `git clone`)

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
grad-cam
captum
gdown
```

See `requirements.txt` / `pyproject.toml` for the exact version floors used for training.

## 5. Dataset documentation

See [`DATA.md`](DATA.md) for the official dataset URL, dataset version, data split, preprocessing procedure, and the exact script to reproduce the data used in the experiments — required for the course submission.

## 6. How to run

1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```
2. Create an environment and install dependencies — either works, pick whichever you already use:
   ```bash
   # with pip
   pip install -r requirements.txt

   # or with uv (uses the pinned .python-version / uv.lock)
   uv sync
   ```
3. Download the dataset (run inside `notebooks/data_exploration.ipynb` or a dedicated script in `src/datasets/`):
   ```python
   from datasets import load_dataset
   ds = load_dataset("moondream/glaucoma-detection")
   ```
4. Train a model: open the corresponding notebook (`train_cnn_scratch.ipynb`, `train_densenet121.ipynb`, `train_resnet50.ipynb`) on Colab/Kaggle — each notebook's first cell `git clone`s this repo and `%cd`s into it, so `src/` is importable right away — and run the training loop.
5. Save checkpoints to `checkpoints/` and results/metrics to `results/metrics/`.
6. To reproduce the XAI results: open `notebooks/xai_analysis.ipynb` on Colab, run the setup cell (clones the repo), then the checkpoint-download cell (pulls from the Google Drive link via `gdown`), then run the rest to apply LayerCAM + GradCAM++ + Occlusion and save the comparison grids to `results/figures/xai/`.

## 7. Demo for the presentation

For the live demo, do **not** rely on downloading the full dataset on the spot — classroom Wi-Fi can be unreliable.

Instead:
- Put 5–10 representative fundus images in `demo_samples/`. This folder **is** committed to git, so it's immediately available right after `git clone`, on any machine, with no Hugging Face/internet dependency.
- Point the demo script/notebook at `demo_samples/` instead of the full dataset when showing predictions + XAI heatmaps live.
- Test the demo once on a freshly cloned copy of the repo beforehand, to confirm it runs without needing the full dataset or an internet connection.

## 8. Collaboration conventions

- Everyone works on their own branch (`feature/resnet50`, `feature/densenet121`, `feature/cnn-scratch`, `feature/xai-<method>`, ...) and merges into `main` via Pull Request.
- Do not commit the raw dataset or model checkpoints (.pt/.pth) — these stay in `.gitignore` and are pulled on demand instead: the dataset via `datasets.load_dataset()`, checkpoints via Google Drive (see `notebooks/xai_analysis.ipynb` for the `gdown`-based download pattern). Only `demo_samples/` (a handful of sample images) and `result/` folder **are** committed to git, so the team can review them directly on GitHub.
- Shared code (models, datasets, training loop, heatmap plotting functions) lives in `src/` to avoid copy-pasting between each person's notebook.
- Results comparing the 3 models (accuracy, F1, confusion matrix, etc.) are consolidated into a shared file in `results/metrics/` so the whole team can review it when choosing the best model.
