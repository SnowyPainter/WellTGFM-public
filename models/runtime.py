"""Explicit runtime profile; importing models never changes global precision."""
import json
from pathlib import Path
import torch

PROFILE = Path(__file__).resolve().parents[1] / 'configs/rtx_pro_6000.json'


def configure_runtime(device='cuda:0'):
    profile = json.loads(PROFILE.read_text())
    profile['device'] = str(device)
    torch.set_num_threads(profile['cpu_threads'])
    if torch.device(device).type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; use --device cpu')
        capability = torch.cuda.get_device_capability(device)
        if capability >= (12, 0) and 'sm_120' not in torch.cuda.get_arch_list():
            raise RuntimeError('Blackwell requires a PyTorch build supporting sm_120')
        torch.set_float32_matmul_precision(profile['float32_matmul_precision'])
    profile['effective_matmul_precision'] = torch.get_float32_matmul_precision()
    return profile
