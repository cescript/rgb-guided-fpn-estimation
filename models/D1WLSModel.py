import torch
from utility.GetDevice import GetDevice

# get D1WLS implementations
from .d1wls.D1WLSNetwork import D1WLSNetwork

class D1WLSModel:
    """
        1D-Weighted-Least-Square-Destriping-for-Uncooled-Infrared-Images
    """
    # constructor for the D1WLS model
    def __init__(self):
        # get the device
        self.device = GetDevice

        # define the D1WLS model using the D1WLSDESTRIPEDEMO parameters
        self.d1wls_model = D1WLSNetwork(lamda=40, titer=3).to(self.device)

        # no gradient will be required for this model
        for param in self.d1wls_model.parameters():
            param.requires_grad = False
        self.d1wls_model.eval()
    
    # generate an fpn tensor from the given beta
    @staticmethod
    def fpn_from_beta(beta_est):
        return torch.stack((torch.ones_like(beta_est), -beta_est), dim=0)
    
    # takes noisy ir and rgb images and return fpn_estimation
    def forward(self, irn_imgs, rgb_imgs):
        # get the beta for each image
        beta_imgs = [self.d1wls_model.forward(irn_imgs[idx].squeeze()) for idx in range(len(irn_imgs))]

        # now we have fpn_est, create fpn_imgs list
        return [self.fpn_from_beta(beta_imgs[idx]) for idx in range(len(irn_imgs))]
    
                