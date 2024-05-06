import numpy as np
import random
import torch


def set_seed(seed: int = 42):
    """
    Set the random seed for various libraries to ensure reproducibility.

    Parameter:
        seed (int, optional): The seed value to set. Defaults to 42.

    This function follows the guide provided by the official PyTorch documentation:
    https://pytorch.org/docs/stable/notes/randomness.html (accessed: 21.06.2023)
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(_):
    """
    This function is used by the torch.utils.data.DataLoader().
    It assures a code reproducibility

    The code of this function is based on and in parts taken from the official PyTorch documentation:
    https://pytorch.org/docs/stable/notes/randomness.html Section: DataLoader (accessed: 21.06.2023)
    """

    worker_seed = 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)
