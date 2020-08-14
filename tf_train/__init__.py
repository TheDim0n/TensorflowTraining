from tensorflow.keras import layers, losses, metrics, models, optimizers
from tensorflow import config

from .train_loop import *
from .utils import *


__version__ = "1.3.0"

def check_GPU():
    GPU = config.list_logical_devices('GPU')
    CPU = config.list_logical_devices('CPU')
    DEVICE = GPU[0].name if GPU else CPU[0].name
    print("Enabled device:", DEVICE)
