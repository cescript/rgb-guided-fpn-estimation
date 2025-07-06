import os
import torch
import torchvision
from torchvision.utils import save_image

class OutputVisualizer:
    
    def __init__(self, path, subfolder, save_logs=True):
        self.output_directory = os.path.join(path, subfolder)
        self.visuals_directory = os.path.join(self.output_directory, "visuals")
        self.log_name = os.path.join(self.output_directory, "logs.txt")

        # create output directory for the images and logs
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        
        # now make visuals directory
        if not os.path.exists(self.visuals_directory):
            os.makedirs(self.visuals_directory)
            
        # check the log file and delete if exist
        if save_logs and os.path.isfile(self.log_name):
            os.remove(self.log_name)
    
    # assume that each image has 1 x H x W
    def save_denoising_results(self, epoch, rgb_img, irc_img, irn_img, ire_img):
        
        # make the image array
        combined = torch.cat([rgb_img, irc_img.repeat(3,1,1), irn_img.repeat(3,1,1), ire_img.repeat(3,1,1)], dim=2)

        # save images to the disk
        img_path = os.path.join(self.visuals_directory, 'img_ep%.3d.png' % (epoch))
        save_image(combined, img_path)

    # assume that each image has 1 x H x W
    def save_inference_results(self, model_name, image_idx, rgb_img, irc_img, irn_img, ire_img):
        # make the image array
        combined = torch.cat([rgb_img, irc_img.repeat(3, 1, 1), irn_img.repeat(3, 1, 1), ire_img.repeat(3, 1, 1)], dim=2)

        # save images to the disk
        img_path = os.path.join(self.visuals_directory, f"img_{model_name}_{image_idx}.png")
        save_image(combined, img_path)
        
    # normalize the given FPN channel to [0,1] range assuming it is in [off-scl, off+scl] range
    def normalize_fpn(self, fpn_channel, scalar_offset):
        a = scalar_offset[1] - scalar_offset[0]
        b = scalar_offset[1] + scalar_offset[0]
        return ((fpn_channel - a) / (b - a)).clamp(0,1)
    
    # assume that each image has 2 x H x W
    def save_fpn_results(self, model_name, image_idx, fpn_img, fpe_img):
        # make the image array
        combined = torch.cat([
            # from OutputNoiseScaler we know alpha,beta ranges
            torch.cat([self.normalize_fpn(fpn_img[0,:,:], [0.2, 1.0]), self.normalize_fpn(fpe_img[0,:,:], [0.2, 1.0])], dim=0), # alpha channel
            torch.cat([self.normalize_fpn(fpn_img[1,:,:], [0.2, 0.0]), self.normalize_fpn(fpe_img[1,:,:], [0.2, 0.0])], dim=0), # beta channel
        ], dim=1).unsqueeze(0)

        # save images to the disk
        img_path = os.path.join(self.visuals_directory, f"fpn_{model_name}_{image_idx}.png")
        save_image(combined, img_path)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, total_epochs, duration, losses):
        # print the log and goto next epoch
        message = '[epoch %d / %d] completed in %d sec' % (epoch, total_epochs, duration)
        message += '\tlosses: '
        for k, v in losses.items():
            message += '%s: %.8f ' % (k, v)
        
        # print message to stdout and file
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

        