import os
import torch
from utility.GetDevice import GetDevice

# main implementation of the algorithm
from .multiview.MULTIVIEWNetwork import MULTIVIEWNetwork

class MULTIVIEWModel:
    """
        Fixed Pattern Noise Removal For Multi-View Single-Sensor Infrared Camera (WACV 2024)
    """
    # constructor for the MULTIVIEW-FPN model
    def __init__(self):
        # get rgb multiplier as float
        self.device = GetDevice()

        # define the MULTIVIEW-FPN network models
        self.multiview_model = MULTIVIEWNetwork(self.device)

        # no gradient will be required for this model
        for param in self.multiview_model.parameters():
            param.requires_grad = False
        self.multiview_model.eval()

    # generate an fpn tensor from the given beta
    @staticmethod
    def fpn_from_beta(beta_est):
        return torch.cat([torch.ones_like(beta_est), -beta_est], dim=0)

    # takes noisy ir and rgb images and return fpn_estimation
    def forward(self, irn_imgs, rgb_imgs):
        
        # multiview expects [AGGx1xHxW] inputs, irn is a list of 1xHxW tensors
        beta_est = self.multiview_model(torch.stack(irn_imgs, dim=0))

        # now we have fpn_est, create fpn_imgs list
        return [self.fpn_from_beta(beta_est) for _ in range(len(irn_imgs))]
    
    
    
                