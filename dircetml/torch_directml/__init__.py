import os
import platform

# import torch to load in directml
import torch

# Load the directml dll into the process
platform = 'win' if platform.system() == 'Windows' else 'linux'
if platform == 'win':
    directml_dll = os.path.join(os.path.dirname(__file__), 'DirectML.dll')
else:
    directml_dll = os.path.join(os.path.dirname(__file__), 'libdirectml.so')
torch.ops.load_library(directml_dll)

# import native apis
import torch_directml_native

from .device import *