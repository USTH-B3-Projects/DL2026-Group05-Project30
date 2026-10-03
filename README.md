# Explainable Glaucoma Stage Classification

Classify glaucoma severity from retinal fundus images, then apply XAI (Explainable AI) methods to visualize and explain which regions drive the model's predictions.

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
| Nhật & Việt | DenseNet121 | Transfer learning, fine-tuning |
| An & Ngân Anh | CNN from scratch | Custom architecture |

> Once the 3 models are trained and compared, the team will meet again to finalize the **3 XAI methods** and assign pairs to each one (2 people per method). This section will be updated once that's decided.

## 3. Repository structure

```
.
├── data/                       # Raw data is NOT committed to git (see .gitignore)
│   ├── raw/                    # Original dataset downloaded from Hugging Face
│   └── processed/              # Preprocessed data (if any)
│
├── notebooks/                  # Notebooks run on Colab/Kaggle, kept for reference/reproducibility
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_cnn_scratch.ipynb
│   ├── 03_train_densenet121.ipynb
│   ├── 04_train_resnet50.ipynb
│   └── 05_xai_analysis.ipynb
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

## 4. Libraries used

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

## 5. How to run

1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```
2. Create an environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset (run inside `notebooks/01_data_exploration.ipynb` or a dedicated script in `src/datasets/`):
   ```python
   from datasets import load_dataset
   ds = load_dataset("moondream/glaucoma-detection")
   ```
4. Train a model: open the corresponding notebook (`02_train_cnn_scratch.ipynb`, `03_train_densenet121.ipynb`, `04_train_resnet50.ipynb`) on Colab/Kaggle, import modules from `src/` (mount/clone the repo inside the notebook), and run the training loop.
5. Save checkpoints to `checkpoints/` and results/metrics to `results/`.
6. Once the best model is selected, run `05_xai_analysis.ipynb` to apply the 3 XAI methods and save the heatmaps to `results/figures/`.

## 6. Collaboration conventions

- Everyone works on their own branch (`feature/resnet50`, `feature/densenet121`, `feature/cnn-scratch`, `feature/xai-<method>`, ...) and merges into `main` via Pull Request.
- Do not commit the raw dataset, model checkpoints (.pt/.pth), or other large files directly — add them to `.gitignore` and share via Drive/Kaggle Dataset if needed.
- Shared code (models, datasets, training loop, heatmap plotting functions) lives in `src/` to avoid copy-pasting between each person's notebook.
- Results comparing the 3 models (accuracy, F1, confusion matrix, etc.) are consolidated into a shared file in `results/metrics/` so the whole team can review it when choosing the best model.
