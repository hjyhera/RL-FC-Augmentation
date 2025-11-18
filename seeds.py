# seeds.py (예시)
import os, random, numpy as np, torch

_INITIALIZED = False

def setup_determinism(seed: int):
    global _INITIALIZED
    if _INITIALIZED:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA 결정성
    # torch import 이후:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)  # warn_only 인자 없이 (구버전 호환)
    except TypeError:
        torch.use_deterministic_algorithms(True)

    # TF32 끄면 수치 일관성↑(선택)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    _INITIALIZED = True
