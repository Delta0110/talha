"""
Model değerlendirme betiği / Model evaluation script
"""

import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.talha.model import SimpleNeuralNetwork
from src.talha.data_loader import load_data
from src.talha.trainer import Trainer
from src.talha.config import MODEL_CONFIG, SAVED_MODELS_DIR


def parse_args():
    """
    Komut satırı argümanlarını ayrıştır
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Model Değerlendirme")
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="model_final.pth",
        help="Model dosyasının yolu"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch boyutu"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Sonuçları görselleştir"
    )
    
    return parser.parse_args()


def visualize_predictions(model, test_loader, device, num_images=10):
    """
    Tahminleri görselleştir / Visualize predictions
    
    Args:
        model: Değerlendirilecek model / Model to evaluate
        test_loader: Test veri yükleyici / Test data loader
        device: Cihaz / Device
        num_images (int): Görselleştirilecek görüntü sayısı
    """
    model.eval()
    
    # İlk batch'i al / Get first batch
    data, targets = next(iter(test_loader))
    data, targets = data.to(device), targets.to(device)
    
    # Tahmin yap / Make predictions
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
    
    # Görselleştirme / Visualization
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(data))):
        img = data[i].cpu().squeeze()
        true_label = targets[i].item()
        pred_label = predicted[i].item()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Gerçek: {true_label}\nTahmin: {pred_label}', 
                         color='green' if true_label == pred_label else 'red')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png')
    print("Tahminler 'predictions_visualization.png' dosyasına kaydedildi")
    plt.close()


def plot_confusion_matrix(model, test_loader, device, num_classes=10):
    """
    Karışıklık matrisi oluştur / Create confusion matrix
    
    Args:
        model: Değerlendirilecek model
        test_loader: Test veri yükleyici
        device: Cihaz
        num_classes (int): Sınıf sayısı
    """
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            for t, p in zip(targets, predicted):
                confusion_matrix[t.item()][p.item()] += 1
    
    # Görselleştirme / Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_matrix, cmap='Blues')
    
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xlabel('Tahmin Edilen / Predicted')
    ax.set_ylabel('Gerçek / True')
    ax.set_title('Karışıklık Matrisi / Confusion Matrix')
    
    # Değerleri göster / Show values
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j, i, confusion_matrix[i, j],
                          ha="center", va="center", color="black")
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Karışıklık matrisi 'confusion_matrix.png' dosyasına kaydedildi")
    plt.close()


def main():
    """
    Ana değerlendirme fonksiyonu / Main evaluation function
    """
    args = parse_args()
    
    print("=" * 60)
    print("Model Değerlendirme / Model Evaluation")
    print("=" * 60)
    
    # Cihaz ayarı / Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCihaz: {device}")
    
    # Veri yükleyicileri oluştur / Create data loaders
    print("\nVeri yükleniyor...")
    _, _, test_loader = load_data(batch_size=args.batch_size)
    print(f"Test örnekleri: {len(test_loader.dataset)}")
    
    # Model oluştur / Create model
    print("\nModel yükleniyor...")
    model = SimpleNeuralNetwork(
        input_size=MODEL_CONFIG["input_size"],
        hidden_sizes=MODEL_CONFIG["hidden_sizes"],
        output_size=MODEL_CONFIG["output_size"],
        dropout_rate=MODEL_CONFIG["dropout_rate"]
    )
    
    # Trainer oluştur ve modeli yükle / Create trainer and load model
    trainer = Trainer(model)
    try:
        trainer.load_model(args.model_path)
    except Exception as e:
        print(f"Model yüklenirken hata: {e}")
        print(f"Lütfen geçerli bir model dosyası belirtin: {SAVED_MODELS_DIR}")
        return
    
    # Test seti üzerinde değerlendirme / Evaluate on test set
    print("\nModel değerlendiriliyor...")
    test_loss, test_acc = trainer.validate(test_loader)
    
    print(f"\nTest Sonuçları:")
    print(f"  Kayıp / Loss: {test_loss:.4f}")
    print(f"  Doğruluk / Accuracy: {test_acc:.2f}%")
    
    # Görselleştirme / Visualization
    if args.visualize:
        print("\nGörselleştirmeler oluşturuluyor...")
        visualize_predictions(model, test_loader, device)
        plot_confusion_matrix(model, test_loader, device)
    
    print("\n" + "=" * 60)
    print("Değerlendirme tamamlandı! / Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
