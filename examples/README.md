# Örnekler / Examples

Bu dizin, Talha projesinin çeşitli kullanım örneklerini içerir.

This directory contains various usage examples for the Talha project.

## Örnekler Listesi / Example List

### 1. simple_example.py
En basit kullanım örneği - sentetik veri ile model eğitimi

The simplest usage example - model training with synthetic data

**Çalıştırma / Run:**
```bash
python examples/simple_example.py
```

**İçerik / Contents:**
- Sentetik veri oluşturma / Creating synthetic data
- Basit model oluşturma / Creating a simple model
- Eğitim / Training
- Değerlendirme / Evaluation
- Tahmin yapma / Making predictions

### 2. custom_model_example.py
Özel model mimarileri ile çalışma örneği

Example of working with custom model architectures

**Çalıştırma / Run:**
```bash
python examples/custom_model_example.py
```

**İçerik / Contents:**
- Konvolüsyonel sinir ağı modeli / Convolutional neural network model
- Derin model mimarisi / Deep model architecture
- Özel modelleri Trainer ile kullanma / Using custom models with Trainer
- Batch normalization ve dropout kullanımı / Using batch normalization and dropout

## Tüm Örnekleri Çalıştırma / Running All Examples

```bash
# Basit örnek / Simple example
python examples/simple_example.py

# Özel model örneği / Custom model example
python examples/custom_model_example.py
```

## Notlar / Notes

- Tüm örnekler sentetik veri kullanır, bu nedenle hızlı çalışır
- Gerçek veri setleri için ana eğitim scriptlerini kullanın (`train.py`)
- Örnekler eğitim amaçlıdır ve production kullanımı için optimize edilmemiştir

---

- All examples use synthetic data, so they run quickly
- For real datasets, use the main training scripts (`train.py`)
- Examples are for educational purposes and not optimized for production use
