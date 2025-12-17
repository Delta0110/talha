"""
Basit kullanım örneği / Simple usage example
"""

import torch
from pathlib import Path
import sys

# Proje kök dizinini ekle / Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.talha import (
    SimpleNeuralNetwork,
    Trainer,
    set_seed
)
from src.talha.data_loader import create_synthetic_data


def main():
    """
    En basit kullanım örneği - sentetik veri ile
    Simplest usage example - with synthetic data
    """
    print("=" * 60)
    print("Basit Kullanım Örneği / Simple Usage Example")
    print("=" * 60)
    
    # 1. Rastgelelik kontrolü / Reproducibility
    set_seed(42)
    
    # 2. Sentetik veri oluştur / Create synthetic data
    print("\n1. Veri oluşturuluyor... / Creating data...")
    train_loader, _ = create_synthetic_data(num_samples=500, input_size=784)
    val_loader, _ = create_synthetic_data(num_samples=100, input_size=784)
    print("   ✓ Veri hazır / Data ready")
    
    # 3. Model oluştur / Create model
    print("\n2. Model oluşturuluyor... / Creating model...")
    model = SimpleNeuralNetwork(
        input_size=784,
        hidden_sizes=[64, 32],
        output_size=10,
        dropout_rate=0.1
    )
    print(f"   ✓ Model oluşturuldu / Model created")
    print(f"   Parametreler / Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Eğitim yapılandırması / Training configuration
    config = {
        "num_epochs": 3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "device": "cpu"
    }
    
    # 5. Trainer oluştur / Create trainer
    print("\n3. Trainer oluşturuluyor... / Creating trainer...")
    trainer = Trainer(model, config=config)
    print(f"   ✓ Trainer hazır / Trainer ready")
    print(f"   Cihaz / Device: {trainer.device}")
    
    # 6. Eğitim / Training
    print("\n4. Eğitim başlıyor... / Starting training...")
    trainer.train(train_loader, val_loader, num_epochs=config["num_epochs"])
    
    # 7. Değerlendirme / Evaluation
    print("\n5. Değerlendirme... / Evaluation...")
    val_loss, val_acc = trainer.validate(val_loader)
    print(f"   Doğrulama Kaybı / Validation Loss: {val_loss:.4f}")
    print(f"   Doğrulama Doğruluğu / Validation Accuracy: {val_acc:.2f}%")
    
    # 8. Tahmin örneği / Prediction example
    print("\n6. Örnek tahmin... / Sample prediction...")
    sample_data = next(iter(val_loader))[0][:5]  # İlk 5 örnek
    predictions = model.predict(sample_data)
    print(f"   İlk 5 tahmin / First 5 predictions: {predictions.tolist()}")
    
    print("\n" + "=" * 60)
    print("✓ Örnek tamamlandı! / Example completed!")
    print("=" * 60)
    
    return trainer, model


if __name__ == "__main__":
    trainer, model = main()
