"""
Ana eğitim betiği / Main training script
"""

import torch
import argparse
from pathlib import Path

from src.talha.model import SimpleNeuralNetwork
from src.talha.data_loader import load_data
from src.talha.trainer import Trainer
from src.talha.config import TRAINING_CONFIG, MODEL_CONFIG


def parse_args():
    """
    Komut satırı argümanlarını ayrıştır
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Talha AI Eğitim Projesi")
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAINING_CONFIG["batch_size"],
        help="Batch boyutu (varsayılan: 32)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_CONFIG["num_epochs"],
        help="Epoch sayısı (varsayılan: 10)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=TRAINING_CONFIG["learning_rate"],
        help="Öğrenme oranı (varsayılan: 0.001)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=TRAINING_CONFIG["device"],
        choices=["cpu", "cuda"],
        help="Eğitim cihazı (varsayılan: cpu)"
    )
    
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Veri setini otomatik indirmeyi devre dışı bırak / Disable automatic dataset download"
    )
    
    return parser.parse_args()


def main():
    """
    Ana eğitim fonksiyonu / Main training function
    """
    # Argümanları ayrıştır / Parse arguments
    args = parse_args()
    
    # Yapılandırmayı güncelle / Update configuration
    config = TRAINING_CONFIG.copy()
    config["batch_size"] = args.batch_size
    config["num_epochs"] = args.epochs
    config["learning_rate"] = args.lr
    config["device"] = args.device
    
    print("=" * 60)
    print("Talha - Python Yapay Zeka Eğitim Projesi")
    print("=" * 60)
    print(f"\nEğitim Yapılandırması:")
    print(f"  Batch Boyutu: {config['batch_size']}")
    print(f"  Epoch Sayısı: {config['num_epochs']}")
    print(f"  Öğrenme Oranı: {config['learning_rate']}")
    print(f"  Cihaz: {config['device']}")
    print()
    
    # Veri yükleyicileri oluştur / Create data loaders
    print("Veri yükleniyor... / Loading data...")
    try:
        train_loader, val_loader, test_loader = load_data(batch_size=config["batch_size"])
        print(f"Eğitim örnekleri: {len(train_loader.dataset)}")
        print(f"Doğrulama örnekleri: {len(val_loader.dataset)}")
        print(f"Test örnekleri: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        print("Not: İlk çalıştırmada MNIST veri seti otomatik olarak indirilecektir.")
        return
    
    # Model oluştur / Create model
    print("\nModel oluşturuluyor... / Creating model...")
    model = SimpleNeuralNetwork(
        input_size=MODEL_CONFIG["input_size"],
        hidden_sizes=MODEL_CONFIG["hidden_sizes"],
        output_size=MODEL_CONFIG["output_size"],
        dropout_rate=MODEL_CONFIG["dropout_rate"]
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametreler: {total_params:,}")
    print(f"Eğitilebilir parametreler: {trainable_params:,}")
    
    # Trainer oluştur / Create trainer
    trainer = Trainer(model, config=config)
    
    # Eğitimi başlat / Start training
    trainer.train(train_loader, val_loader, num_epochs=config["num_epochs"])
    
    # Final modeli kaydet / Save final model
    print("\nFinal model kaydediliyor...")
    trainer.save_model("model_final.pth")
    trainer.save_history("training_history.json")
    
    # Test seti üzerinde değerlendirme / Evaluate on test set
    print("\nTest seti üzerinde değerlendirme yapılıyor...")
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"Test Kaybı: {test_loss:.4f}")
    print(f"Test Doğruluğu: {test_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("Eğitim başarıyla tamamlandı! / Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
