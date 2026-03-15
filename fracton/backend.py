"""
Fracton tensor backend — device management and tensor construction.

PyTorch is the sole tensor backend. CUDA is used automatically when available.
Scalar math (constants, fibonacci, number_theory) uses Python's math module.
Field operations, projections, and SEC evolution use torch tensors.

Usage:
    from fracton.backend import get_device, to_tensor, default_dtype

    device = get_device()  # cuda if available, else cpu
    t = to_tensor([[1, 2], [3, 4]])  # on best available device
"""

import torch


def get_device() -> torch.device:
    """Return the best available compute device.

    Returns cuda if a CUDA-capable GPU is available, otherwise cpu.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def default_dtype() -> torch.dtype:
    """Return the default dtype for physics computations.

    float64 for the precision required by physical constant derivations.
    """
    return torch.float64


def to_tensor(
    data,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Convert data to a torch tensor on the best available device.

    Args:
        data: Input data (list, tuple, number, or existing tensor).
        dtype: Optional dtype override. Defaults to float64.
        device: Optional device override. Defaults to get_device().

    Returns:
        torch.Tensor on the specified (or best) device.
    """
    if dtype is None:
        dtype = default_dtype()
    if device is None:
        device = get_device()

    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    return torch.tensor(data, dtype=dtype, device=device)
