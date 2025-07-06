import torch
import torch.nn as nn

class OLSNoiseEstimator(nn.Module):
    """
    Estimate FPN using OLS algorithm using the input as formed [BATCH x 1 x HEIGHT x WIDTH]
    """
    def __init__(self):
        super(OLSNoiseEstimator, self).__init__()
        self.irn = []
        self.irc = []

    # embedding must be reset for different fpn
    def reset_embedding(self):
        self.irn = []
        self.irc = []
    
    def forward(self, irc_image, irn_image):
        # get the images into list
        self.irn.append(irn_image)
        self.irc.append(irc_image)
        
        # wait until all images are given
        if len(self.irn) < 1:
            return []
        
        # now we have all images, solve equation x = a.z + b size: [B, 1, H, W]
        x = torch.stack(self.irc, dim=1).squeeze(dim=2)  # B x AGG x H x W
        z = torch.stack(self.irn, dim=1).squeeze(dim=2)  # B x AGG x H x W

        # get x and z means
        mx = x.mean(dim=1).unsqueeze(dim=1)  # B x 1 x H x W
        mz = z.mean(dim=1).unsqueeze(dim=1)  # B x 1 x H x W
        
        # find fraction parts for alpha
        eps = 1e-8
        alpha_num = ((z - mz) * (x - mx)).sum(dim=1)  # B x H x W
        alpha_den = ((z - mz) * (z - mz)).sum(dim=1)  # B x H x W

        # now calculate the alpha : B x 1 x H x W
        alpha_s = (alpha_num + eps) / (alpha_den + eps)
        alpha_s = alpha_s.unsqueeze(dim=1)
        
        # find beta : B x 1 x H x W
        beta_s = mx - alpha_s * mz
        
        # concat alpha and beta
        fpn_est = torch.cat([alpha_s, beta_s], dim=1)
        return fpn_est