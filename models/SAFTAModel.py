import os
import torch

# get safta implementation
from models.safta.SaftaDenoiser import SAFTADenoiser
from models.safta.SaftaNoiseEstimator import SaftaNoiseEstimator
from models.safta.OLSNoiseEstimator import OLSNoiseEstimator
from utility.GetDevice import GetDevice

class SAFTAModel:
    """
        Self Attended Feature Temporal Aggregation Module for IR-FPA denoising
    """
    # constructor for the safta model
    def __init__(self, use_rgb, fpn_estimator:str):
        
        # get rgb multiplier as float
        self.device = GetDevice()
        self.rgb_multiplier = torch.tensor(1.0 if use_rgb else 0.0, device=self.device)
        
        # create denoiser and fpn estimator
        self.denoiser = SAFTADenoiser(is_train_mode=False).to(self.device)
        self.denoiser.load_state(os.path.join("models", "safta", "weights", "best_denoiser.pth"))
        
        # pick the fpn estimator
        if fpn_estimator == "OLS":
            self.fpn_estimator = OLSNoiseEstimator()
        elif fpn_estimator == "GRU":
            self.fpn_estimator = SaftaNoiseEstimator(is_train_mode=False).to(self.device)
            self.fpn_estimator.load_state(os.path.join("models", "safta", "weights", "best_fpn_estimator_normalized.pth"))
        else:
            print(f"invalid fpn estimator")
            exit(1)
    
    # takes noisy ir and rgb images and return fpn_estimation
    def forward(self, irn_imgs, rgb_imgs):
        # get the number of images in sequence
        aggregation_size = len(irn_imgs)
        assert len(irn_imgs) == len(rgb_imgs) and aggregation_size >= 2, f"invalid input: noisy and rgb images should have at least three images"
        
        # get the ire_imgs Kx1xHxW
        ire_imgs = self.denoiser.forward(self.rgb_multiplier * torch.stack(rgb_imgs, dim=0), torch.stack(irn_imgs, dim=0))
        
        # now feed ire_imgs with irn_imgs to fpn estimator network
        fpn_est = None
        self.fpn_estimator.reset_embedding()
        for idx in range(aggregation_size):
            fpn_est = self.fpn_estimator.forward(ire_imgs[idx].unsqueeze(0), irn_imgs[idx].unsqueeze(0))
        
        # make fpn_est a list
        fpn_imgs = [fpn_est.squeeze(0) for _ in range(aggregation_size)]
        
        # now we have fpn_est
        return fpn_imgs
    
    
    
                
