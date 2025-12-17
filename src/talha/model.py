"""
Sinir ağı modeli tanımı / Neural network model definition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNeuralNetwork(nn.Module):
    """
    Basit bir ileri beslemeli sinir ağı modeli
    Simple feedforward neural network model
    """
    
    def __init__(self, input_size=784, hidden_sizes=None, output_size=10, dropout_rate=0.2):
        """
        Args:
            input_size (int): Giriş boyutu / Input size
            hidden_sizes (list): Gizli katman boyutları / Hidden layer sizes
            output_size (int): Çıkış boyutu (sınıf sayısı) / Output size (number of classes)
            dropout_rate (float): Dropout oranı / Dropout rate
        """
        super(SimpleNeuralNetwork, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Katmanları oluştur / Create layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Çıkış katmanı / Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        İleri geçiş / Forward pass
        
        Args:
            x (torch.Tensor): Giriş tensörü / Input tensor
            
        Returns:
            torch.Tensor: Çıkış tensörü / Output tensor
        """
        # Giriş düzleştirme / Flatten input
        x = x.view(-1, self.input_size)
        return self.network(x)
    
    def predict(self, x):
        """
        Tahmin yap / Make prediction
        
        Args:
            x (torch.Tensor): Giriş tensörü / Input tensor
            
        Returns:
            torch.Tensor: Tahmin edilen sınıflar / Predicted classes
        """
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
            return predicted
