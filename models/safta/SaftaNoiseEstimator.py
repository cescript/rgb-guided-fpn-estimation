import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# import critical components of the SAFTA
from models.safta.blocks import ResidualConvBlock, FixedPatternNoiseDecoder, LowHighFrequencyLoss, SimpleConvGRUCell

class SaftaNoiseEstimator(nn.Module):
    def __init__(self, learning_rate=0.0001, decay_start_epoch = 100, total_epochs=100, is_train_mode=False):
        super().__init__()
        
        # set self params
        self.gru_hidden_channel = 32
        self.feature_size = 32
        self.alpha_scalar_offset = [0.2, 1.0]
        self.beta_scalar_offset = [0.2, 0.0]
        
        # create the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # create sub-units of the SAFTA in GPU
        self.encoder = nn.Sequential(
            ResidualConvBlock(2, self.feature_size//2),
            ResidualConvBlock(self.feature_size//2, self.feature_size),
            ResidualConvBlock(self.feature_size, self.feature_size)).to(self.device)
        
        # create a simple gated-memory unit
        self.convgru = SimpleConvGRUCell(self.feature_size, self.gru_hidden_channel).to(self.device)
        
        # create decoders as a basic units
        self.fpn_alpha_decoder = FixedPatternNoiseDecoder(self.gru_hidden_channel, self.alpha_scalar_offset).to(self.device)
        self.fpn_beta_decoder = FixedPatternNoiseDecoder(self.gru_hidden_channel, self.beta_scalar_offset).to(self.device)

        # set embedding to None at start
        self.fpn_embedding = None
        self.total_loss = 0.0
        
        # create classes that only needed in training mode
        self.is_train_mode = is_train_mode
        if is_train_mode:
            # set the loss function
            self.loss_function = LowHighFrequencyLoss(loss_lh_weights=(0.6, 0.4)).to(self.device)
            
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
    
    # embedding must be reset for different fpn
    def reset_embedding(self):
        self.fpn_embedding = None
        self.total_loss = 0

    def forward(self, irc_image, irn_image):
        # get the shape
        batch_size,_,height,width = irc_image.shape
        
        # now we have clan and noisy image, estimate the FPN
        if self.fpn_embedding is None:
            self.fpn_embedding = torch.zeros(batch_size, self.gru_hidden_channel, height, width, device=self.device)
        
        # get the fused features [B, 3, H, W] --> [B, 128, H, W]
        fused_input = torch.cat([irc_image, irn_image], dim=1)
        fused_feat = self.encoder(fused_input)
        
        # get the new embedding
        self.fpn_embedding = self.convgru(fused_feat, self.fpn_embedding)

        # decode current embedding and estimate fpn
        alpha_s = self.fpn_alpha_decoder(self.fpn_embedding)
        beta_s = self.fpn_beta_decoder(self.fpn_embedding)
        
        # fpn and clean ir estimates
        fpn_est = torch.cat([alpha_s, beta_s], dim=1)

        # return the estimations
        return fpn_est
    
    def accumulate_loss(self, irc_img, irn_image, fpn_est, weight):
        # split fpn into alpha,beta
        alpha_s, beta_s = torch.chunk(fpn_est, chunks=2, dim=1)
        irc_est = alpha_s * irn_image + beta_s
        self.total_loss += weight * self.loss_function(output=irc_est, target=irc_img)
        return irc_est
    
    # calculate the loss using the inputs and optimize network
    def optimize_parameters(self, ):
        self.optimizer.zero_grad()
        self.total_loss.backward()
        self.optimizer.step()
        
        # return the total_loss loss
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
        checkpoint = torch.load(model_name)
        self.load_state_dict(checkpoint['model_state_dict'])
        if self.is_train_mode:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.fpn_embedding = None
    
    def calculate_loss(self, output, target):
        # ensure input and target are of the same shape
        if output.shape != target.shape:
            raise ValueError("output and target must have the same shape")
    
        # compute the squared differences Bx2xHxW
        diff = output - target
        squared_diff = diff ** 2
    
        # apply weights
        weighted_squared_diff = squared_diff * self.loss_ab_weights
    
        # compute the mean over all dimensions except for the channel dimension
        loss = torch.sqrt(weighted_squared_diff.mean(dim=[0, 2, 3]))
    
        # sum over the channel dimensions to get the final loss value
        return loss.sum()


