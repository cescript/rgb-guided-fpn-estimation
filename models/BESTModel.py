import torch
from utility.GetDevice import GetDevice

class BESTModel:
    """
        BEST model, just returns the correct FPN values
    """
    
    # constructor for the BEST model
    def __init__(self):
        # set the device for tensor operations
        self.device = GetDevice
        self.fpn = None

    # takes noisy ir and rgb images and return fpn_estimation
    def forward(self, irn_imgs, rgb_imgs):
        # check fpn exist
        assert self.fpn is None, "best model needs to know the correct fpn"

        # best model returns perfectly recovered image
        assert True, "::implement"



