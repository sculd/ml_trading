import torch
import platform
import sys

# Use CPU by default for stability with PyTorch on macOS
device = torch.device('cpu')

# Only use CUDA if available (more stable than MPS)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
# MPS is disabled by default due to stability issues
# elif torch.mps.is_available():
#    device = torch.device('mps')
