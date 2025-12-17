# Talha - Python Yapay Zeka Eğitim Projesi

Bu proje, Python tabanlı yapay zeka ve makine öğrenmesi eğitimi için oluşturulmuş kapsamlı bir başlangıç projesidir.

This is a comprehensive starter project for Python-based artificial intelligence and machine learning training.

## 🎯 Özellikler / Features

- ✅ PyTorch tabanlı derin öğrenme / PyTorch-based deep learning
- ✅ MNIST veri seti ile örnek uygulama / Example application with MNIST dataset
- ✅ Modüler ve genişletilebilir yapı / Modular and extensible structure
- ✅ Eğitim, doğrulama ve test pipeline'ı / Training, validation and test pipeline
- ✅ Model kaydetme ve yükleme / Model saving and loading
- ✅ Eğitim geçmişi takibi / Training history tracking
- ✅ Jupyter Notebook örnekleri / Jupyter Notebook examples
- ✅ Komut satırı araçları / Command-line tools

## 📁 Proje Yapısı / Project Structure

```
talha/
├── src/
│   └── talha/
│       ├── __init__.py          # Paket başlatma / Package initialization
│       ├── config.py            # Yapılandırma ayarları / Configuration settings
│       ├── model.py             # Sinir ağı modeli / Neural network model
│       ├── data_loader.py       # Veri yükleme / Data loading
│       └── trainer.py           # Eğitim sınıfı / Training class
├── data/
│   ├── raw/                     # Ham veri / Raw data
│   └── processed/               # İşlenmiş veri / Processed data
├── models/
│   └── saved/                   # Kaydedilmiş modeller / Saved models
├── logs/                        # Eğitim logları / Training logs
├── notebooks/
│   └── 01_basic_training.ipynb  # Örnek notebook / Example notebook
├── train.py                     # Ana eğitim betiği / Main training script
├── evaluate.py                  # Değerlendirme betiği / Evaluation script
├── requirements.txt             # Python bağımlılıkları / Python dependencies
├── setup.py                     # Kurulum dosyası / Setup file
└── README.md                    # Bu dosya / This file
```

## 🚀 Kurulum / Installation

### 1. Depoyu Klonlayın / Clone the Repository

```bash
git clone https://github.com/Delta0110/talha.git
cd talha
```

### 2. Sanal Ortam Oluşturun / Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin / Install Dependencies

```bash
pip install -r requirements.txt

# veya proje paketini kurun / or install the project package
pip install -e .
```

## 💻 Kullanım / Usage

### Komut Satırından Eğitim / Training from Command Line

Temel kullanım:
```bash
python train.py
```

Özelleştirilmiş parametrelerle:
```bash
python train.py --epochs 20 --batch-size 64 --lr 0.0001 --device cuda
```

Parametreler / Parameters:
- `--epochs`: Epoch sayısı (varsayılan: 10)
- `--batch-size`: Batch boyutu (varsayılan: 32)
- `--lr`: Öğrenme oranı (varsayılan: 0.001)
- `--device`: Eğitim cihazı - cpu veya cuda (varsayılan: cpu)

### Model Değerlendirme / Model Evaluation

```bash
python evaluate.py --model-path models/saved/model_final.pth --visualize
```

### Jupyter Notebook ile Kullanım / Using Jupyter Notebook

```bash
jupyter notebook notebooks/01_basic_training.ipynb
```

### Python Kodu İçinde Kullanım / Using in Python Code

```python
from src.talha import SimpleNeuralNetwork, load_data, Trainer

# Veri yükle / Load data
train_loader, val_loader, test_loader = load_data(batch_size=32)

# Model oluştur / Create model
model = SimpleNeuralNetwork(
    input_size=784,
    hidden_sizes=[128, 64],
    output_size=10
)

# Eğitim / Training
trainer = Trainer(model)
trainer.train(train_loader, val_loader, num_epochs=10)

# Model kaydet / Save model
trainer.save_model("my_model.pth")
```

## 📊 Veri Seti / Dataset

Proje, varsayılan olarak MNIST el yazısı rakam veri setini kullanır. İlk çalıştırmada otomatik olarak indirilir.

The project uses the MNIST handwritten digits dataset by default. It will be downloaded automatically on first run.

- Eğitim örnekleri / Training samples: 54,000
- Doğrulama örnekleri / Validation samples: 6,000
- Test örnekleri / Test samples: 10,000
- Görüntü boyutu / Image size: 28x28 piksel
- Sınıf sayısı / Number of classes: 10 (0-9 rakamları)

## 🧠 Model Mimarisi / Model Architecture

Varsayılan model mimarisi:

```
Input (784) → Linear (128) → ReLU → Dropout(0.2) 
            → Linear (64) → ReLU → Dropout(0.2)
            → Linear (10) → Output
```

## 📈 Sonuçlar / Results

Tipik eğitim sonuçları:
- Eğitim doğruluğu / Training accuracy: ~98%
- Test doğruluğu / Test accuracy: ~97%
- Eğitim süresi / Training time: ~5-10 dakika (CPU)

## 🔧 Yapılandırma / Configuration

Eğitim parametreleri `src/talha/config.py` dosyasında özelleştirilebilir:

```python
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 10,
    "random_seed": 42,
}

MODEL_CONFIG = {
    "input_size": 784,
    "hidden_sizes": [128, 64],
    "output_size": 10,
    "dropout_rate": 0.2,
}
```

## 🛠️ Geliştirme / Development

### Testler / Tests

```bash
# Testleri çalıştır (yakında eklenecek)
pytest tests/
```

### Kod Stili / Code Style

```bash
# Kod formatla
black src/
flake8 src/
```

## 📚 Kaynaklar / Resources

- [PyTorch Dokümantasyonu](https://pytorch.org/docs/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Python Machine Learning](https://www.python.org/)

## 🤝 Katkıda Bulunma / Contributing

Katkılarınızı bekliyoruz! Lütfen:
1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📝 Lisans / License

Bu proje MIT lisansı altında lisanslanmıştır.

## 👤 Yazar / Author

**Delta0110**

## 📧 İletişim / Contact

Sorularınız için issue açabilirsiniz.

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
