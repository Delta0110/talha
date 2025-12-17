"""
Yardımcı fonksiyonlar / Utility functions
"""

import torch
import random
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """
    Rastgelelik için seed ayarla / Set seed for reproducibility
    
    Args:
        seed (int): Seed değeri / Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Model parametrelerini say / Count model parameters
    
    Args:
        model: PyTorch modeli / PyTorch model
        
    Returns:
        dict: Parametre istatistikleri / Parameter statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params
    }


def save_config(config, filepath):
    """
    Yapılandırmayı JSON dosyasına kaydet
    Save configuration to JSON file
    
    Args:
        config (dict): Yapılandırma sözlüğü
        filepath (str or Path): Dosya yolu
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def load_config(filepath):
    """
    JSON dosyasından yapılandırma yükle
    Load configuration from JSON file
    
    Args:
        filepath (str or Path): Dosya yolu
        
    Returns:
        dict: Yapılandırma sözlüğü
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_training_history(history, save_path=None):
    """
    Eğitim geçmişini görselleştir
    Visualize training history
    
    Args:
        history (dict): Eğitim geçmişi
        save_path (str or Path, optional): Kayıt yolu
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Kayıp grafiği / Loss plot
    if 'train_loss' in history and history['train_loss']:
        ax1.plot(history['train_loss'], label='Eğitim Kaybı / Train Loss', marker='o')
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(history['val_loss'], label='Doğrulama Kaybı / Val Loss', marker='s')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Kayıp / Loss')
    ax1.set_title('Model Kaybı / Model Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Doğruluk grafiği / Accuracy plot
    if 'train_accuracy' in history and history['train_accuracy']:
        ax2.plot(history['train_accuracy'], label='Eğitim Doğruluğu / Train Acc', marker='o')
    if 'val_accuracy' in history and history['val_accuracy']:
        ax2.plot(history['val_accuracy'], label='Doğrulama Doğruluğu / Val Acc', marker='s')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Doğruluk (%) / Accuracy (%)')
    ax2.set_title('Model Doğruluğu / Model Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafik kaydedildi: {save_path}")
    
    plt.show()


def format_time(seconds):
    """
    Saniyeyi okunabilir formata çevir
    Convert seconds to readable format
    
    Args:
        seconds (float): Saniye cinsinden süre
        
    Returns:
        str: Formatlanmış süre
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def get_device(prefer_cuda=True):
    """
    Uygun cihazı al / Get available device
    
    Args:
        prefer_cuda (bool): CUDA tercih et
        
    Returns:
        torch.device: Cihaz
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA kullanılıyor: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CPU kullanılıyor")
    
    return device


def print_model_summary(model, input_size=(1, 784)):
    """
    Model özetini yazdır
    Print model summary
    
    Args:
        model: PyTorch modeli
        input_size (tuple): Giriş boyutu
    """
    print("=" * 80)
    print("MODEL ÖZETİ / MODEL SUMMARY")
    print("=" * 80)
    print(f"\n{model}\n")
    
    params = count_parameters(model)
    print(f"Toplam Parametreler / Total Parameters: {params['total']:,}")
    print(f"Eğitilebilir Parametreler / Trainable Parameters: {params['trainable']:,}")
    print(f"Eğitilemez Parametreler / Non-trainable Parameters: {params['non_trainable']:,}")
    
    # Parametre boyutu
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)
    print(f"\nModel Boyutu / Model Size: {total_size_mb:.2f} MB")
    print("=" * 80)
