"""
Eğitim yapılandırması / Training configuration
"""

import os
from pathlib import Path

# Proje dizinleri / Project directories
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"
LOGS_DIR = ROOT_DIR / "logs"

# Eğitim parametreleri / Training parameters
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 10,
    "train_split": 0.8,
    "validation_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42,
    "device": "cpu",  # "cuda" veya "cpu"
}

# Model parametreleri / Model parameters
MODEL_CONFIG = {
    "input_size": 784,  # Örnek: MNIST için 28x28
    "hidden_sizes": [128, 64],
    "output_size": 10,
    "dropout_rate": 0.2,
}

# Kayıt parametreleri / Logging parameters
LOGGING_CONFIG = {
    "log_interval": 10,  # Her kaç batch'te bir log yazdırılacak
    "save_interval": 5,  # Her kaç epoch'ta bir model kaydedilecek
}

def get_config():
    """
    Tüm yapılandırma ayarlarını döndürür
    Returns all configuration settings
    """
    return {
        "training": TRAINING_CONFIG,
        "model": MODEL_CONFIG,
        "logging": LOGGING_CONFIG,
        "directories": {
            "root": ROOT_DIR,
            "data": DATA_DIR,
            "raw_data": RAW_DATA_DIR,
            "processed_data": PROCESSED_DATA_DIR,
            "models": MODELS_DIR,
            "saved_models": SAVED_MODELS_DIR,
            "logs": LOGS_DIR,
        }
    }
