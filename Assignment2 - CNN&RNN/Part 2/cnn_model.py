from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

  def __init__(self, n_channels, n_classes):
    """
    Initializes CNN object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """
    super(CNN, self).__init__()
    
    self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(128)
    self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(256)
    self.bn4 = nn.BatchNorm2d(256)
    self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.bn5 = nn.BatchNorm2d(512)
    self.bn6 = nn.BatchNorm2d(512)
    self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.bn7 = nn.BatchNorm2d(512)
    self.bn8 = nn.BatchNorm2d(512)
    self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    self.fc = nn.Linear(512, n_classes)

  def forward(self, x):
    """
    Performs forward pass of the input.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """
    x = F.relu(self.bn1(self.conv1(x)))
    x = self.pool1(x)
    
    x = F.relu(self.bn2(self.conv2(x)))
    x = self.pool2(x)
    
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(x)))
    x = self.pool3(x)
    
    x = F.relu(self.bn5(self.conv5(x)))
    x = F.relu(self.bn6(self.conv6(x)))
    x = self.pool4(x)
    
    x = F.relu(self.bn7(self.conv7(x)))
    x = F.relu(self.bn8(self.conv8(x)))
    x = self.pool5(x)
    
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    
    return x
