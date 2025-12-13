import torch
import torch.nn as nn
import torch.optim as optim
from utility.GetDevice import GetDevice
from torch.optim.lr_scheduler import LambdaLR

# import critical components of the SAFTA
from models.safta.blocks import LowHighFrequencyLoss

class SAFTADenoiser(nn.Module):
    def __init__(self, learning_rate=0.0001, decay_start_epoch = 100, total_epochs=100, is_train_mode=False):
        super().__init__()
        
        # set the device type
        self.device = GetDevice()
        
        # set base channel for encoders
        bc = 32
        self.pre_conv1 = nn.Sequential(nn.Conv2d( 4, bc, kernel_size=3, stride=1, padding=1), nn.ReLU()).to(self.device)
        self.pre_conv2 = nn.Sequential(nn.Conv2d(bc, bc, kernel_size=3, stride=1, padding=1), nn.ReLU()).to(self.device)
        self.pre_conv3 = nn.Sequential(nn.Conv2d(bc, bc, kernel_size=3, stride=1, padding=1), nn.ReLU()).to(self.device)

        # simple residual computation blocks
        self.res_conv1 = nn.Sequential(nn.Conv2d(bc, bc, kernel_size=3, stride=1, padding=1), nn.ReLU()).to(self.device)
        self.res_conv2 = nn.Sequential(nn.Conv2d(bc, bc, kernel_size=3, stride=1, padding=1), nn.ReLU()).to(self.device)
        self.res_conv3 = nn.Sequential(nn.Conv2d(bc, 1, kernel_size=3, stride=1, padding=1)).to(self.device)

        # make special path for stripe noise
        self.stripe_conv1 = nn.Sequential(nn.Conv2d(bc, bc, kernel_size=(1, 9), padding=(0, 4)), nn.ReLU()).to(self.device)
        self.stripe_conv2 = nn.Sequential(nn.Conv2d(bc, bc, kernel_size=(1, 9), padding=(0, 4)), nn.ReLU()).to(self.device)
        self.stripe_conv3 = nn.Sequential(nn.Conv2d(bc, 1, kernel_size=1)).to(self.device)

        # ENCODER BLOCK: enc_conv + downsample + enc_conv + downsample ...
        self.enc_conv1 = nn.Sequential(nn.Conv2d(bc * 1, bc * 1, kernel_size=3, stride=1, padding=1), nn.ReLU()).to(self.device)
        self.enc_conv2 = nn.Sequential(nn.Conv2d(bc * 1, bc * 1, kernel_size=3, stride=1, padding=1), nn.ReLU()).to(self.device)
        self.enc_conv3 = nn.Sequential(nn.Conv2d(bc * 1, bc * 2, kernel_size=3, stride=1, padding=1), nn.ReLU()).to(self.device)
        self.enc_conv4 = nn.Sequential(nn.Conv2d(bc * 2, bc * 2, kernel_size=3, stride=1, padding=1), nn.ReLU()).to(self.device)
        self.enc_conv5 = nn.Sequential(nn.Conv2d(bc * 2, bc * 3, kernel_size=3, stride=1, padding=1), nn.ReLU()).to(self.device)

        # generic down/up sample layers (used many times)
        self.downsample = nn.MaxPool2d(2).to(self.device)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True).to(self.device)

        # DECODER BLOCK: dec_conv + upsample + dec_conv + upsample ...
        self.dec_conv5 = nn.Sequential(nn.Conv2d(bc * 5, bc * 3, kernel_size=3, padding=1), nn.ReLU()).to(self.device) # [e5+e4]
        self.dec_conv4 = nn.Sequential(nn.Conv2d(bc * 5, bc * 3, kernel_size=3, padding=1), nn.ReLU()).to(self.device) # [p+e3]
        self.dec_conv3 = nn.Sequential(nn.Conv2d(bc * 4, bc * 2, kernel_size=3, padding=1), nn.ReLU()).to(self.device) # [p+e2]
        self.dec_conv2 = nn.Sequential(nn.Conv2d(bc * 3, bc * 2, kernel_size=3, padding=1), nn.ReLU()).to(self.device) # [p+e1]
        self.dec_conv1 = nn.Sequential(nn.Conv2d(bc * 2, bc * 1, kernel_size=3, padding=1), nn.ReLU()).to(self.device) # [p]

        # create classes that only needed in training mode
        self.is_train_mode = is_train_mode
        if self.is_train_mode:
            self.loss_function = LowHighFrequencyLoss(loss_lh_weights=(0.1, 0.9)).to(self.device)
            
            # create optimizer
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999))

            # learning rate scheduler with linear decay after a warm-up period
            def lambda_rule(epoch):
                lr_factor = 1.0 - max(0, epoch - decay_start_epoch) / float(total_epochs - decay_start_epoch + 1)
                return max(lr_factor, 0.0)
            
            # create the actual schedular
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        # make the network ready for inference
        else:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def normalize_input(self, net_in):
        # net_in: [B, C, H, W]
        mu = net_in.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        std = net_in.std(dim=(2, 3), keepdim=True) + 1e-6

        # calculate normalized input
        return (net_in - mu) / std, std

    # forward process
    def forward(self, rgb_image, irn_image):
        # [BATCH x 1 x HEIGHT x WIDTH]
        # make an input vector using both features
        net_in = torch.cat([rgb_image, irn_image], dim=1)

        # INPUT BLOCK
        net_in_norm, std = self.normalize_input(net_in)
        a1 = self.pre_conv1(net_in_norm)
        a2 = self.pre_conv2(a1)
        a3 = self.pre_conv2(a2)

        # ENCODER BLOCK for RGB GUIDANCE
        e1 = self.enc_conv1(a3)
        p1 = self.downsample(e1)
        e2 = self.enc_conv2(p1)
        p2 = self.downsample(e2)
        e3 = self.enc_conv3(p2)
        p3 = self.downsample(e3)
        e4 = self.enc_conv4(p3)
        p4 = self.downsample(e4)
        e5 = self.enc_conv5(p4)
    
        # DECODER BLOCK for RGB GUIDANCE
        d5 = self.upsample(e5)
        d4 = self.dec_conv5(torch.cat([d5, e4], dim=1))
        d4 = self.upsample(d4)
        d3 = self.dec_conv4(torch.cat([d4, e3], dim=1))
        d3 = self.upsample(d3)
        d2 = self.dec_conv3(torch.cat([d3, e2], dim=1))
        d2 = self.upsample(d2)
        d1 = self.dec_conv2(torch.cat([d2, e1], dim=1))
        d0 = self.dec_conv1(d1)

        # STRIPE LEARNING
        s0 = d0.mean(dim=2, keepdim=True)
        s1 = self.stripe_conv1(s0)
        s2 = self.stripe_conv2(s1)
        s3 = self.stripe_conv3(s2)

        # broadcast over height
        ss = s3.expand(-1, -1, irn_image.size(2), -1)

        # RESIDUAL LEARNING
        z1 = self.res_conv1(d0)
        z2 = self.res_conv2(z1)
        rs = self.res_conv3(z2)

        # RESIDUAL to IMAGE
        clean = irn_image + (rs + ss) * std[:, 3:4, :, :]

        # clamp result in test mode
        if not self.training:
            clean = torch.clamp(clean, 0.0, 1.0)

        # return the current estimates for the clean image
        return clean

    # calculate the loss and do parameter update
    def optimize_parameters(self, irc_img, irc_est):
        self.total_loss = self.loss_function(irc_est, irc_img)
        self.optimizer.zero_grad()
        self.total_loss.backward()
        self.optimizer.step()
        
        # return the current loss
        return self.total_loss
        
    def step_scheduler(self):
        self.scheduler.step()

    # save the state of the model to checkpoint
    def save_state(self, checkpoint):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, checkpoint)
    
    # load the state of the model from checkpoint
    def load_state(self, model_name):
        checkpoint = torch.load(model_name, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if self.is_train_mode:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

