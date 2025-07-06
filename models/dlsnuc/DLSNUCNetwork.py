import torch
import torch.nn as nn
import numpy as np

class DLSNUCNetwork(nn.Module):
    """Create DLS-NUC Convolutional Network"""
    
    def __init__(self, model):
        super(DLSNUCNetwork, self).__init__()
        
        # load the model parameters
        weights = model['weight'][0, 0]   # 1x11 cell
        biasses = model['bias'][0, 0]     # 1x11 cell
        
        # single Conv2D + Normalization + ReLU block
        self.ld1 = nn.Conv2d(1,  32, kernel_size=3, stride=1, padding=1)
        self.initialize(self.ld1, weights[0, 0], biasses[0, 0])
        
        # create max pooling layer
        self.mp0 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        
        # create a list of layers
        layers = []
        for idx in range(1, 8):
            cl = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
            self.initialize(cl, weights[0, idx], biasses[0, idx])
            layers.append(cl)
            layers.append(nn.ReLU())
        # create single sequence of layers from layers list
        self.layers = nn.Sequential(*layers)

        # create the last layers
        self.ld9  = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.initialize(self.ld9, weights[0, 8], biasses[0, 8])
        
        self.ld10 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=3)
        self.initialize(self.ld10, weights[0, 9], biasses[0, 9])
        
        self.ld11 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.initialize(self.ld11, np.expand_dims(weights[0, 10], axis=3), biasses[0, 10])
    
    # initialize the given weights to the layer
    def initialize(self, layer, weights, biasses):
        layer.weight.data = torch.Tensor(weights).permute(3, 2, 0, 1)
        layer.bias.data = torch.Tensor(biasses).squeeze(1)
        
    def forward(self, inImage):
        
        convfea = self.ld1.forward(inImage)
        convfea1 = convfea
        convfea = self.mp0.forward(convfea)
        
        # apply conv2d + ReLU stack
        convfea = self.layers.forward(convfea)
        
        convfea9  = self.ld9.forward(convfea)
        convfea10 = self.ld10.forward(convfea9)
        convfea = torch.cat((convfea1, convfea10), dim=1)
        convfea = self.ld11.forward(convfea)
        
        return convfea