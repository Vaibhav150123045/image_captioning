import random
import numpy as np
import torch


def get_device(device_id):
    return torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")


def make_reproducible(seed=0):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
