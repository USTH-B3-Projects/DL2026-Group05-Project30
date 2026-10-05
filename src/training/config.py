import random
import numpy as np
import torch

from src.datasets.glaucoma_dataset import IDX_TO_CLASS

# ---- Paths ----
DATA_DIR = "data/raw"
CHECKPOINT_DIR = "checkpoints"
RESULTS_METRICS_DIR = "results/metrics"
RESULTS_FIGURES_DIR = "results/figures"

# ---- Dataset ----
HF_DATASET_NAME = "moondream/glaucoma-detection"
IMAGE_SIZE = 224          # resize target (DenseNet121/ResNet50 pretrained expect 224x224)
NUM_CLASSES = 3
CLASS_NAMES = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

# ---- Training ----
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

# ---- Device ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = SEED) -> None:
    """Call this at the top of every notebook before building the model/dataloaders,
    so results are reproducible and comparable across the 3 models."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    