"""
Özel model örneği / Custom model example
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Proje kök dizinini ekle / Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.talha import Trainer, set_seed
from src.talha.data_loader import create_synthetic_data


class ConvolutionalModel(nn.Module):
    """
    Özel konvolüsyonel sinir ağı modeli
    Custom convolutional neural network model
    """
    
    def __init__(self, num_classes=10):
        super(ConvolutionalModel, self).__init__()
        
        # Konvolüsyon katmanları / Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Giriş şekillendirme / Reshape input
        x = x.view(-1, 1, 28, 28)
        
        # Konvolüsyon bloğu 1 / Convolution block 1
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        
        # Konvolüsyon bloğu 2 / Convolution block 2
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Düzleştir / Flatten
        x = x.view(-1, 32 * 7 * 7)
        
        # Fully connected katmanlar / Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        """
        Tahmin yap / Make prediction
        """
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
            return predicted


class DeepModel(nn.Module):
    """
    Daha derin bir model örneği
    Example of a deeper model
    """
    
    def __init__(self, input_size=784, num_classes=10):
        super(DeepModel, self).__init__()
        
        self.input_size = input_size
        
        # Derin mimari / Deep architecture
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        return self.layers(x)
    
    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
            return predicted


def train_custom_model(model, model_name="custom"):
    """
    Özel modeli eğit / Train custom model
    """
    print(f"\n{'=' * 60}")
    print(f"{model_name} Modeli Eğitimi / {model_name} Model Training")
    print(f"{'=' * 60}")
    
    # Veri oluştur / Create data
    print("\nVeri oluşturuluyor... / Creating data...")
    train_loader, _ = create_synthetic_data(num_samples=500, input_size=784)
    val_loader, _ = create_synthetic_data(num_samples=100, input_size=784)
    
    # Model bilgisi / Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametreleri / Model parameters: {total_params:,}")
    
    # Eğitim yapılandırması / Training configuration
    config = {
        "num_epochs": 5,
        "learning_rate": 0.001,
        "device": "cpu"
    }
    
    # Trainer oluştur ve eğit / Create trainer and train
    trainer = Trainer(model, config=config)
    trainer.train(train_loader, val_loader, num_epochs=config["num_epochs"])
    
    # Değerlendirme / Evaluation
    val_loss, val_acc = trainer.validate(val_loader)
    print(f"\nSonuç / Result:")
    print(f"  Doğruluk / Accuracy: {val_acc:.2f}%")
    
    return trainer


def main():
    """
    Ana fonksiyon / Main function
    """
    print("=" * 60)
    print("Özel Model Örnekleri / Custom Model Examples")
    print("=" * 60)
    
    set_seed(42)
    
    # 1. Konvolüsyonel model / Convolutional model
    print("\n1. Konvolüsyonel Model / Convolutional Model")
    conv_model = ConvolutionalModel(num_classes=10)
    trainer1 = train_custom_model(conv_model, "Konvolüsyonel / Convolutional")
    
    # 2. Derin model / Deep model
    print("\n2. Derin Model / Deep Model")
    deep_model = DeepModel(input_size=784, num_classes=10)
    trainer2 = train_custom_model(deep_model, "Derin / Deep")
    
    print("\n" + "=" * 60)
    print("✓ Tüm örnekler tamamlandı! / All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
