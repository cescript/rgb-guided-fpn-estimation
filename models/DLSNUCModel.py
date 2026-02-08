import os
import torch
import scipy.io as sio
from utility.GetDevice import GetDevice

# import actual network
from .dlsnuc.DLSNUCNetwork import DLSNUCNetwork

class DLSNUCModel:
    """
        PyTorch implementation of DLS-NUC Model
        Model Parameters and utils folder fetched from https://github.com/hezw2016/DLS-NUC/tree/master
        verifyPyTorchOutputs.m function is used to verify the PyTorch model using the outputs of the Matlab model
        Total absolute difference between two implementation is about 0.005, which is 6x10^{-8} per pixel on average
    """
    
    # constructor for the DLS-NUC model
    def __init__(self):
        # get the device
        self.device = GetDevice()
    
        # define the DLS-NUC network models
        mcn = sio.loadmat("models/dlsnuc/original/model1.mat", squeeze_me=False)
        self.dlsnuc_model = DLSNUCNetwork(mcn['model']).to(self.device)

        # no gradient will be required for this model
        for param in self.dlsnuc_model.parameters():
            param.requires_grad = False
        self.dlsnuc_model.eval()

    # generate an fpn tensor from the given beta
    @staticmethod
    def fpn_from_beta(beta_est):
        return torch.cat([torch.ones_like(beta_est), -beta_est], dim=0)

    # takes noisy ir and rgb images and return fpn_estimation
    def forward(self, irn_imgs, rgb_imgs):
        # get the beta for each image
        dlsnuc_input = torch.stack(irn_imgs, dim=0)
        beta_imgs = self.dlsnuc_model.forward(dlsnuc_input)
    
        # now we have fpn_est, create fpn_imgs list
        return [self.fpn_from_beta(beta_imgs[idx]) for idx in range(len(irn_imgs))]
