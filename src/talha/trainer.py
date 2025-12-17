"""
Model eğitim sınıfı / Model training class
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

from .config import TRAINING_CONFIG, LOGGING_CONFIG, SAVED_MODELS_DIR, LOGS_DIR


class Trainer:
    """
    Model eğitimi için yardımcı sınıf
    Helper class for model training
    """
    
    def __init__(self, model, config=None):
        """
        Args:
            model: Eğitilecek model / Model to train
            config (dict): Eğitim yapılandırması / Training configuration
        """
        self.model = model
        self.config = config if config else TRAINING_CONFIG
        
        # Cihaz ayarı / Device setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.get("device") == "cuda" 
            else "cpu"
        )
        self.model.to(self.device)
        
        # Kayıp fonksiyonu ve optimizer / Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 0.001)
        )
        
        # Eğitim geçmişi / Training history
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
    
    def train_epoch(self, train_loader):
        """
        Bir epoch eğitim / Train one epoch
        
        Args:
            train_loader: Eğitim veri yükleyici / Training data loader
            
        Returns:
            tuple: (ortalama_kayıp, doğruluk) / (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Eğitim")):
            data, target = data.to(self.device), target.to(self.device)
            
            # Gradyanları sıfırla / Zero gradients
            self.optimizer.zero_grad()
            
            # İleri geçiş / Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Geri yayılım / Backward pass
            loss.backward()
            self.optimizer.step()
            
            # İstatistikleri güncelle / Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """
        Doğrulama / Validation
        
        Args:
            val_loader: Doğrulama veri yükleyici / Validation data loader
            
        Returns:
            tuple: (ortalama_kayıp, doğruluk) / (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Doğrulama"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = running_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """
        Tam eğitim döngüsü / Full training loop
        
        Args:
            train_loader: Eğitim veri yükleyici / Training data loader
            val_loader: Doğrulama veri yükleyici / Validation data loader
            num_epochs (int): Epoch sayısı / Number of epochs
        """
        if num_epochs is None:
            num_epochs = self.config.get("num_epochs", 10)
        
        print(f"\nEğitim başlıyor... / Training starting...")
        print(f"Cihaz: {self.device}")
        print(f"Epoch sayısı: {num_epochs}\n")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Eğitim / Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Doğrulama / Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Geçmişi güncelle / Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            
            # Sonuçları yazdır / Print results
            print(f"Eğitim Kaybı: {train_loss:.4f}, Doğruluk: {train_acc:.2f}%")
            print(f"Doğrulama Kaybı: {val_loss:.4f}, Doğruluk: {val_acc:.2f}%\n")
            
            # Model kaydetme / Save model
            if (epoch + 1) % LOGGING_CONFIG.get("save_interval", 5) == 0:
                self.save_model(f"model_epoch_{epoch + 1}.pth")
        
        print("Eğitim tamamlandı! / Training completed!")
    
    def save_model(self, filename):
        """
        Modeli kaydet / Save model
        
        Args:
            filename (str): Dosya adı / Filename
        """
        save_path = SAVED_MODELS_DIR / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config,
        }, save_path)
        
        print(f"Model kaydedildi: {save_path}")
    
    def load_model(self, filename):
        """
        Modeli yükle / Load model
        
        Args:
            filename (str): Dosya adı / Filename
        """
        load_path = SAVED_MODELS_DIR / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Model yüklendi: {load_path}")
    
    def save_history(self, filename="training_history.json"):
        """
        Eğitim geçmişini kaydet / Save training history
        
        Args:
            filename (str): Dosya adı / Filename
        """
        save_path = LOGS_DIR / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        print(f"Eğitim geçmişi kaydedildi: {save_path}")
