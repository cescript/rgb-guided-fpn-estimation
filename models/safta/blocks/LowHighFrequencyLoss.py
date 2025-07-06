import torch
import torch.nn as nn
import torch.nn.functional as F

class LowHighFrequencyLoss(nn.Module):
    def __init__(self, loss_lh_weights=(0.5, 0.5), loss_function=nn.L1Loss(reduction='none')):
        super().__init__()
        self.loss_lh_weights = loss_lh_weights
        self.loss_function = loss_function

    def forward(self, output, target):
        # low frequency loss (downsampled)
        loss_scaler = float(output.shape[2] * output.shape[3] / (11 * 11))
        lp_out = F.interpolate(output, size=(11, 11), mode='bicubic', align_corners=False)
        lp_tgt = F.interpolate(target, size=(11, 11), mode='bicubic', align_corners=False)
        lf_loss = self.loss_function(lp_out, lp_tgt) * loss_scaler

        # high frequency loss (original resolution)
        hf_loss = self.loss_function(output, target)
        
        # sum all except batch dimension
        lf_loss = lf_loss.view(lf_loss.size(0), -1).sum(dim=1).mean()
        hf_loss = hf_loss.view(hf_loss.size(0), -1).sum(dim=1).mean()
        
        # weighted sum of both losses
        return self.loss_lh_weights[0] * lf_loss + self.loss_lh_weights[1] * hf_loss
