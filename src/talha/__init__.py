"""
Talha - Python tabanlı yapay zeka eğitim projesi
"""

__version__ = "0.1.0"
__author__ = "Delta0110"

from .model import SimpleNeuralNetwork
from .data_loader import load_data, prepare_datasets
from .trainer import Trainer
from .utils import (
    set_seed,
    count_parameters,
    plot_training_history,
    get_device,
    print_model_summary
)

__all__ = [
    "SimpleNeuralNetwork",
    "load_data",
    "prepare_datasets",
    "Trainer",
    "set_seed",
    "count_parameters",
    "plot_training_history",
    "get_device",
    "print_model_summary",
]
