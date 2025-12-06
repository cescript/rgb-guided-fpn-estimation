import torch
from utility.GetDevice import GetDevice

class EMPTYModel:
    """
        EMPTY model, just returns the [alpha=1, beta=0]
    """
    # constructor for the EMPTY model
    def __init__(self):
    
        # set the device for tensor operations
        self.device = GetDevice
        self.fpn = None

    # takes noisy ir and rgb images and return fpn_estimation
    def forward(self, irn_imgs, rgb_imgs):
        # create an empty fpn [2xHxW] at the first run
        if self.fpn is None:
            alpha = torch.ones_like(irn_imgs[0])
            beta = torch.zeros_like(irn_imgs[0])
            self.fpn_est = torch.cat([alpha, beta], dim=0)

        # make fpn_est a list
        fpn_imgs = [self.fpn_est for _ in range(len(irn_imgs))]
        
        # return the result
        return fpn_imgs
    
    
    
                