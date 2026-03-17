import torch, random, numpy as np

def set_seed(seed=42, deterministic=True, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
