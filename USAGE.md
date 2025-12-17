# Talha - Kullanım Kılavuzu / Usage Guide

Bu doküman, Talha projesinin detaylı kullanım kılavuzudur.

This document is a detailed usage guide for the Talha project.

## 📋 İçindekiler / Table of Contents

1. [Kurulum / Installation](#kurulum--installation)
2. [Hızlı Başlangıç / Quick Start](#hızlı-başlangıç--quick-start)
3. [Veri Yükleme / Data Loading](#veri-yükleme--data-loading)
4. [Model Oluşturma / Model Creation](#model-oluşturma--model-creation)
5. [Eğitim / Training](#eğitim--training)
6. [Değerlendirme / Evaluation](#değerlendirme--evaluation)
7. [Özelleştirme / Customization](#özelleştirme--customization)
8. [İleri Düzey Kullanım / Advanced Usage](#i̇leri-düzey-kullanım--advanced-usage)

## Kurulum / Installation

### Önkoşullar / Prerequisites

- Python 3.8 veya üzeri / Python 3.8 or higher
- pip package manager

### Adım 1: Depoyu Klonlayın / Step 1: Clone the Repository

```bash
git clone https://github.com/Delta0110/talha.git
cd talha
```

### Adım 2: Sanal Ortam Oluşturun / Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Adım 3: Bağımlılıkları Yükleyin / Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Adım 4: Projeyi Kurun / Step 4: Install the Project

```bash
pip install -e .
```

## Hızlı Başlangıç / Quick Start

### En Basit Kullanım / Simplest Usage

```bash
python quick_start.py
```

Bu komut:
- Modeli oluşturur / Creates the model
- Veriyi yükler / Loads the data
- 3 epoch eğitim yapar / Trains for 3 epochs
- Modeli kaydeder / Saves the model

### Tam Eğitim / Full Training

```bash
python train.py
```

## Veri Yükleme / Data Loading

### MNIST Veri Seti / MNIST Dataset

```python
from src.talha import load_data

# Veri yükleyicileri oluştur / Create data loaders
train_loader, val_loader, test_loader = load_data(batch_size=32)

# Batch boyutunu özelleştir / Customize batch size
train_loader, val_loader, test_loader = load_data(batch_size=64)
```

### Sentetik Veri / Synthetic Data

Test için sentetik veri oluşturabilirsiniz:

```python
from src.talha.data_loader import create_synthetic_data

data_loader, dataset = create_synthetic_data(
    num_samples=1000,
    input_size=784,
    num_classes=10
)
```

### Özel Veri Seti / Custom Dataset

Kendi veri setinizi kullanmak için:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Veriyi hazırlayın / Prepare your data
X_train = torch.randn(1000, 784)  # 1000 örnek, 784 özellik
y_train = torch.randint(0, 10, (1000,))  # 10 sınıf

# Dataset oluştur / Create dataset
dataset = TensorDataset(X_train, y_train)

# DataLoader oluştur / Create DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Model Oluşturma / Model Creation

### Varsayılan Model / Default Model

```python
from src.talha import SimpleNeuralNetwork

model = SimpleNeuralNetwork()
```

### Özelleştirilmiş Model / Customized Model

```python
model = SimpleNeuralNetwork(
    input_size=784,
    hidden_sizes=[256, 128, 64],  # 3 gizli katman
    output_size=10,
    dropout_rate=0.3
)
```

### Model Bilgileri / Model Information

```python
from src.talha import print_model_summary, count_parameters

# Model özetini göster / Show model summary
print_model_summary(model)

# Parametre sayısını al / Get parameter count
params = count_parameters(model)
print(f"Toplam parametreler: {params['total']:,}")
```

## Eğitim / Training

### Temel Eğitim / Basic Training

```python
from src.talha import Trainer, load_data, SimpleNeuralNetwork

# Model ve veri / Model and data
model = SimpleNeuralNetwork()
train_loader, val_loader, test_loader = load_data()

# Trainer oluştur / Create trainer
trainer = Trainer(model)

# Eğitimi başlat / Start training
trainer.train(train_loader, val_loader, num_epochs=10)
```

### Özelleştirilmiş Eğitim / Customized Training

```python
config = {
    "batch_size": 64,
    "learning_rate": 0.0001,
    "num_epochs": 20,
    "device": "cuda"  # GPU kullanmak için
}

trainer = Trainer(model, config=config)
trainer.train(train_loader, val_loader, num_epochs=config["num_epochs"])
```

### Model Kaydetme / Save Model

```python
# Eğitim sırasında otomatik kayıt / Automatic save during training
trainer.train(train_loader, val_loader)  # Her 5 epoch'ta otomatik kaydeder

# Manuel kayıt / Manual save
trainer.save_model("my_model.pth")
trainer.save_history("my_history.json")
```

### Model Yükleme / Load Model

```python
trainer = Trainer(model)
trainer.load_model("my_model.pth")
```

## Değerlendirme / Evaluation

### Komut Satırından / From Command Line

```bash
# Temel değerlendirme / Basic evaluation
python evaluate.py --model-path models/saved/model_final.pth

# Görselleştirmeli değerlendirme / With visualization
python evaluate.py --model-path models/saved/model_final.pth --visualize
```

### Python Kodundan / From Python Code

```python
from src.talha import Trainer

trainer = Trainer(model)
trainer.load_model("model_final.pth")

# Test seti üzerinde değerlendirme / Evaluate on test set
test_loss, test_acc = trainer.validate(test_loader)
print(f"Test Doğruluğu: {test_acc:.2f}%")
```

### Tahmin Yapma / Making Predictions

```python
import torch

# Tek bir örnek için tahmin / Prediction for single example
model.eval()
with torch.no_grad():
    sample = test_data[0].unsqueeze(0)  # Batch boyutu ekle
    prediction = model.predict(sample)
    print(f"Tahmin: {prediction.item()}")

# Batch tahmin / Batch prediction
predictions = model.predict(test_batch)
```

## Özelleştirme / Customization

### Yapılandırma Dosyası / Configuration File

`config.yaml` dosyasını düzenleyerek ayarları değiştirebilirsiniz:

```yaml
training:
  batch_size: 64
  learning_rate: 0.0001
  num_epochs: 20
  
model:
  hidden_sizes:
    - 256
    - 128
    - 64
  dropout_rate: 0.3
```

### Kendi Modelinizi Oluşturma / Creating Your Own Model

```python
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Trainer ile kullanım / Use with Trainer
custom_model = CustomModel()
trainer = Trainer(custom_model)
```

## İleri Düzey Kullanım / Advanced Usage

### GPU Kullanımı / GPU Usage

```python
import torch

# GPU kontrolü / Check GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    config = {"device": "cuda"}
else:
    config = {"device": "cpu"}

trainer = Trainer(model, config=config)
```

### Eğitim Geçmişini Görselleştirme / Visualize Training History

```python
from src.talha.utils import plot_training_history

# Eğitimden sonra / After training
plot_training_history(trainer.history, save_path="training_plot.png")
```

### Rastgelelik Kontrolü / Reproducibility

```python
from src.talha import set_seed

# Aynı sonuçları almak için / For reproducible results
set_seed(42)
```

### Learning Rate Scheduling

```python
import torch.optim as optim

# Trainer'ı oluşturduktan sonra / After creating trainer
scheduler = optim.lr_scheduler.StepLR(
    trainer.optimizer, 
    step_size=5, 
    gamma=0.1
)

# Eğitim döngüsünde / In training loop
for epoch in range(num_epochs):
    train_loss, train_acc = trainer.train_epoch(train_loader)
    val_loss, val_acc = trainer.validate(val_loader)
    scheduler.step()  # Learning rate'i güncelle
```

### Erken Durdurma / Early Stopping

```python
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in range(num_epochs):
    train_loss, train_acc = trainer.train_epoch(train_loader)
    val_loss, val_acc = trainer.validate(val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        trainer.save_model("best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Erken durdurma / Early stopping")
            break
```

## Jupyter Notebook Kullanımı / Using Jupyter Notebooks

```bash
# Jupyter'ı başlat / Start Jupyter
jupyter notebook

# Veya Jupyter Lab / Or Jupyter Lab
jupyter lab
```

`notebooks/01_basic_training.ipynb` dosyasını açın ve adım adım eğitim yapın.

## Sorun Giderme / Troubleshooting

### CUDA Hatası / CUDA Error

```python
# CPU'ya zorla / Force CPU
config = {"device": "cpu"}
trainer = Trainer(model, config=config)
```

### Bellek Hatası / Memory Error

```python
# Batch boyutunu küçült / Reduce batch size
train_loader, val_loader, test_loader = load_data(batch_size=16)
```

### Veri İndirme Sorunu / Data Download Issue

```python
# Manuel veri yolu / Manual data path
from src.talha.data_loader import load_mnist_data

train_dataset, test_dataset = load_mnist_data(data_dir="/path/to/data")
```

## Ek Kaynaklar / Additional Resources

- [PyTorch Dokümantasyonu](https://pytorch.org/docs/)
- [MNIST Dataset Info](http://yann.lecun.com/exdb/mnist/)
- [GitHub Repository](https://github.com/Delta0110/talha)

## Destek / Support

Sorularınız için GitHub Issues kullanın:
https://github.com/Delta0110/talha/issues
