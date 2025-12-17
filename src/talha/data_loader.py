"""
Veri yükleme ve işleme / Data loading and processing
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path

from .config import RAW_DATA_DIR, TRAINING_CONFIG


def load_mnist_data(data_dir=None):
    """
    MNIST veri setini yükle
    Load MNIST dataset
    
    Args:
        data_dir (str or Path): Veri dizini / Data directory
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR
    
    # Veri dönüşümleri / Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST ortalaması ve std
    ])
    
    # Eğitim ve test veri setlerini yükle / Load train and test datasets
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    return train_dataset, test_dataset


def prepare_datasets(train_dataset, config=None):
    """
    Veri setlerini eğitim ve doğrulama olarak böl
    Split datasets into training and validation
    
    Args:
        train_dataset: Eğitim veri seti / Training dataset
        config (dict): Yapılandırma sözlüğü / Configuration dictionary
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    if config is None:
        config = TRAINING_CONFIG
    
    # Eğitim ve doğrulama bölünmesi / Train-validation split
    train_size = int(config["train_split"] * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["random_seed"])
    )
    
    return train_dataset, val_dataset


def load_data(batch_size=None, data_dir=None):
    """
    Veri yükleyicileri oluştur
    Create data loaders
    
    Args:
        batch_size (int): Batch boyutu / Batch size
        data_dir (str or Path): Veri dizini / Data directory
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = TRAINING_CONFIG["batch_size"]
    
    # Veri setlerini yükle / Load datasets
    train_dataset, test_dataset = load_mnist_data(data_dir)
    train_dataset, val_dataset = prepare_datasets(train_dataset)
    
    # Veri yükleyicileri oluştur / Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader


def create_synthetic_data(num_samples=1000, input_size=784, num_classes=10):
    """
    Test için sentetik veri oluştur
    Create synthetic data for testing
    
    Args:
        num_samples (int): Örnek sayısı / Number of samples
        input_size (int): Giriş boyutu / Input size
        num_classes (int): Sınıf sayısı / Number of classes
        
    Returns:
        tuple: (data_loader, dataset)
    """
    # Rastgele veri oluştur / Generate random data
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return data_loader, dataset
