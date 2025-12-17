"""
Hızlı başlangıç örneği / Quick start example
"""

import torch
from src.talha import (
    SimpleNeuralNetwork,
    load_data,
    Trainer,
    set_seed,
    print_model_summary
)
from src.talha.config import MODEL_CONFIG, TRAINING_CONFIG


def main():
    """
    Hızlı başlangıç örneği
    Quick start example - trains a model for 3 epochs with synthetic data
    """
    print("=" * 60)
    print("Talha - Hızlı Başlangıç Örneği / Quick Start Example")
    print("=" * 60)
    
    # Rastgelelik için seed ayarla / Set seed for reproducibility
    set_seed(42)
    
    # Küçük bir konfigürasyon / Smaller configuration for quick demo
    config = TRAINING_CONFIG.copy()
    config["num_epochs"] = 3
    config["batch_size"] = 64
    
    print("\n1. Model oluşturuluyor... / Creating model...")
    model = SimpleNeuralNetwork(
        input_size=MODEL_CONFIG["input_size"],
        hidden_sizes=[64, 32],  # Daha küçük model
        output_size=MODEL_CONFIG["output_size"],
        dropout_rate=0.1
    )
    
    # Model özetini göster / Show model summary
    print_model_summary(model, input_size=(1, 784))
    
    print("\n2. Veri yükleniyor... / Loading data...")
    print("   (İlk çalıştırmada MNIST indirilecek, bu biraz zaman alabilir)")
    print("   (MNIST will be downloaded on first run, this may take a while)")
    
    try:
        train_loader, val_loader, test_loader = load_data(
            batch_size=config["batch_size"]
        )
        print(f"   ✓ Eğitim örnekleri: {len(train_loader.dataset)}")
        print(f"   ✓ Doğrulama örnekleri: {len(val_loader.dataset)}")
        print(f"   ✓ Test örnekleri: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"   ✗ Veri yükleme hatası: {e}")
        print("\n   Sentetik veri kullanılıyor... / Using synthetic data...")
        from src.talha.data_loader import create_synthetic_data
        train_loader, _ = create_synthetic_data(num_samples=1000)
        val_loader, _ = create_synthetic_data(num_samples=200)
        test_loader, _ = create_synthetic_data(num_samples=200)
    
    print("\n3. Eğitim başlıyor... / Starting training...")
    print(f"   Epoch sayısı: {config['num_epochs']}")
    print(f"   Batch boyutu: {config['batch_size']}")
    print(f"   Öğrenme oranı: {config['learning_rate']}")
    
    trainer = Trainer(model, config=config)
    trainer.train(train_loader, val_loader, num_epochs=config["num_epochs"])
    
    print("\n4. Test değerlendirmesi... / Testing...")
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"   Test Kaybı / Test Loss: {test_loss:.4f}")
    print(f"   Test Doğruluğu / Test Accuracy: {test_acc:.2f}%")
    
    print("\n5. Model kaydediliyor... / Saving model...")
    trainer.save_model("quick_start_model.pth")
    
    print("\n" + "=" * 60)
    print("✓ Hızlı başlangıç tamamlandı! / Quick start completed!")
    print("=" * 60)
    print("\nSonraki adımlar / Next steps:")
    print("  • Tam eğitim için: python train.py")
    print("  • Model değerlendirme: python evaluate.py")
    print("  • Notebook'u deneyin: jupyter notebook notebooks/01_basic_training.ipynb")


if __name__ == "__main__":
    main()
