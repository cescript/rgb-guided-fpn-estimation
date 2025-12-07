import torch
import warnings
from utility.GetDevice import GetDevice
from torchvision.transforms import Compose, ToTensor, Normalize

# import actual network
from .dcgan.options import TestOptions
from .dcgan.model_singalG import DerainCycleGAN

# dcgan uses deprecated features 'pretrained', ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)

class DCGANModel:
    """
        PyTorch implementation of DestripeCycleGAN Model: DestripeCycleGAN: Stripe Simulation CycleGAN for Unsupervised Infrared Image Destriping
        Model Parameters and folder fetched from https://github.com/xdFai/DestripeCycleGAN?tab=readme-ov-file
    """
    
    # constructor for the DCGAN model
    def __init__(self):
        # get the device
        self.device = GetDevice()

        # dcgan uses test options
        parser = TestOptions()
        self.opts = parser.parse()

        # define the DCGAN network models
        self.dcgan_model = DerainCycleGAN(self.opts)
        self.dcgan_model.setgpu()
        self.dcgan_model.resume(self.opts.resume, train=False)

        # no gradient will be required for this model
        for param in self.dcgan_model.parameters():
            param.requires_grad = False
        self.dcgan_model.eval()

        # create data transform
        transforms = [Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        self.transforms = Compose(transforms)

    # generate an fpn tensor from the given beta
    @staticmethod
    def fpn_from_beta(beta_est):
        return torch.cat([torch.ones_like(beta_est), -beta_est], dim=0)

    # takes noisy ir and rgb images and return fpn_estimation
    def forward(self, irn_imgs, rgb_imgs):
        # get the clean image for each image
        dcgan_stack = torch.stack(irn_imgs, dim=0)
        dcgan_input = self.transforms(dcgan_stack.repeat(1, 3, 1, 1))
        clean_imgs = self.dcgan_model.test_forward(dcgan_input, a2b=self.opts.a2b)

        # normalize tensor output to [0,1] range
        clean_imgs = clean_imgs.mean(dim=1, keepdim=True)
        clean_imgs = torch.clamp(clean_imgs, -1., 1.)
        clean_imgs = (clean_imgs + 1) / 2

        # get beta from clean images
        beta_imgs = dcgan_stack - clean_imgs

        # now we have fpn_est, create fpn_imgs list
        return [self.fpn_from_beta(beta_imgs[idx]) for idx in range(len(irn_imgs))]
