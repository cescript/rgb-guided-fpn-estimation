import torch
import torch.nn as nn

class MULTIVIEWNetwork(nn.Module):
    """ Create Multiview-FPN Algorithm Implementation
        The following implementation utilized on the basis of the original python implementation
        https://github.com/centreborelli/multiview-fpn/tree/main
    """
    
    def __init__(self, device):
        super(MULTIVIEWNetwork, self).__init__()
        
        self.device = device

        # suggested parameters in the paper for Multi-view offline FPN estimation diverges for 6 images
        # used parameters from the original implementation
        self.max_iter = 500
        self.step_size = 7 * 10 ** -2
        self.lambda_o = 5 * 10 ** -2
        
    def forward(self, inImage):
        
        # implementation of multiview-fpn algorithm (inputs: AGGREGATED x 1 x CROP_SIZE x CROP_SIZE)
        images = inImage.to(torch.float64)
        input_stack = 255 * torch.squeeze(images, dim=1)
        
        # create output as zero
        output = torch.zeros((input_stack.shape[1], input_stack.shape[2]), device=self.device, requires_grad=True)

        # create adam optimizer
        optimizer = torch.optim.Adam([output], lr=self.step_size)
        
        # block modified from original implementation of Multi_view_GD_offline.py
        for iterations in range(self.max_iter):

            # zero-out all the gradients
            optimizer.zero_grad(set_to_none=True)
            
            # calculate the loss
            loss = self.MultiTVLoss(output, input_stack, self.lambda_o)

            # do back propagation
            loss.backward(retain_graph=False)
            optimizer.step()

        # output FPN is 1 x CROP_SIZE x CROP_SIZE
        return output.detach().to(inImage.dtype).unsqueeze(0) / 255
    
    # borrowed from utils.py of the original implementation
    def l2_square_norm(self, x):
        return (x ** 2).sum()
    
    # re-written
    def TV(self, img, eps=10**-20):
        # get the gradients for batch image
        grad_j = img[:, :, 1:] - img[:, :, :-1]
        grad_i = img[:, 1:, :] - img[:, :-1, :]

        # add zero column and row back to match original dimensions
        grad_j = torch.nn.functional.pad(grad_j, (0, 1, 0, 0))
        grad_i = torch.nn.functional.pad(grad_i, (0, 0, 0, 1))

        # return the total variation
        return torch.sqrt(grad_i ** 2 + grad_j ** 2 + eps).sum()
    
    # borrowed from utils.py Multi_view_GD_offline.py of the original implementation
    def MultiTVLoss(self, O, y, lambda_O):
        loss_reg = self.TV(y - O)
        loss_data = self.l2_square_norm(O)
        loss_value = loss_reg + lambda_O * loss_data
        return loss_value