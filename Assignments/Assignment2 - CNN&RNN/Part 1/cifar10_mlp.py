import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10MLP(nn.Module):
    def __init__(self, num_classes=10, hidden_units=[512, 512, 256, 256], dropout_rate=0.5):
        super(CIFAR10MLP, self).__init__()
        
        input_size = 32 * 32 * 3
        
        layers = []
        in_features = input_size
        
        for hidden_size in hidden_units:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        x = self.classifier(x)
        return x

