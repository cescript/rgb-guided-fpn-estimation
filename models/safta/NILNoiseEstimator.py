import torch
import torch.nn as nn

class NILNoiseEstimator(nn.Module):
    """
    Estimate FPN using simple difference, assume input as formed [BATCH x 1 x HEIGHT x WIDTH]
    """
    def __init__(self):
        super(NILNoiseEstimator, self).__init__()

    # embedding must be reset for different fpn
    def reset_embedding(self):
        pass

    # generate an fpn tensor from the given beta
    @staticmethod
    def fpn_from_beta(beta_est):
        return torch.cat([torch.ones_like(beta_est), -beta_est], dim=1)

    def forward(self, irc_image, irn_image):
        # [BATCH x 1 x HEIGHT x WIDTH]
        beta_est = irn_image - irc_image

        # now we have fpn_est, create fpn_imgs list
        return self.fpn_from_beta(beta_est)