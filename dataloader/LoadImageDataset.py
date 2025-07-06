import os
import json
import random
from PIL import Image
import torch
import torchvision.transforms.functional as tfunction

# my custom dataset
class LoadImageDataset:

    def __init__(self, dataset_config, vh_flip, crop_random, device):
        
        # runtime parameters
        self.vh_flip = vh_flip
        self.crop_random = crop_random
        self.device = device
        
        # assert on error
        config_file = os.path.join("dataloader", "dataset", f"{dataset_config}.json")
        assert os.path.isfile(config_file), f"config file not found at {config_file}"
        
        # try to read configuration file and set variables
        with open(config_file, 'r') as file:
            config = json.load(file)

            # get image paths
            dataset_path = config["dataset_path"]
            self.rgb_paths = self.GetFilePaths(os.path.join(dataset_path, config["rgb_subdirectory"]))
            self.irc_paths = self.GetFilePaths(os.path.join(dataset_path, config["irc_subdirectory"]))

            # assert if the size doesnt match
            assert len(self.rgb_paths) == len(self.irc_paths), "A and B images must be the same length"
            
            # add the necessary configuration options
            self.name = config["dataset_name"]
            self.resize_after_load = config["resize_after_load"]
            self.crop_after_resize = config["crop_after_resize"]
            self.image_count = len(self.rgb_paths)
            
            # log status
            print(f"{self.name} dataset scanned successfully with {self.image_count} unique images")
        
    def __getitem__(self, index):
    
        # read rgb and ir images as pytorch tensors
        RGB = []
        IRC = []
        
        # load images as tensors
        for idx in index:
            rgb = tfunction.to_tensor(Image.open(self.rgb_paths[idx % self.image_count]).convert('RGB'))
            irc = tfunction.to_tensor(Image.open(self.irc_paths[idx % self.image_count]).convert('L'))
            
            # resize after load
            rgb = tfunction.resize(rgb, self.resize_after_load, interpolation=tfunction.InterpolationMode.BILINEAR)
            irc = tfunction.resize(irc, self.resize_after_load, interpolation=tfunction.InterpolationMode.BILINEAR)
            
            RGB.append(rgb)
            IRC.append(irc)
            
        # convert list to tensor and get the cropped patches
        tA, tB = self.GetTransformedAB(RGB, IRC)

        # return RGB and IR pairs
        return tA, tB
    
    def __len__(self):
        return self.image_count

    # return the images inside the given directory
    @staticmethod
    def GetFilePaths(folder):
        image_file_paths = []
        for root, dirs, filenames in os.walk(folder):
            filenames = sorted(filenames)
            for filename in filenames:
                input_path = os.path.abspath(root)
                file_path = os.path.join(input_path, filename)
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    image_file_paths.append(file_path)
        
            break  # prevent descending into subfolders
        return image_file_paths

    def GetTransformedAB(self, A, B):
    
        # convert everything into tensor
        A = torch.stack(A).to(self.device)
        B = torch.stack(B).to(self.device)

        # if crop_size given, random crop the both image
        if self.crop_after_resize[0] >= 0:
            if self.crop_random:
                # crop position for A and B
                x = random.randint(0, max(0, tfunction.get_image_size(A)[0] - self.crop_after_resize[0]))
                y = random.randint(0, max(0, tfunction.get_image_size(A)[1] - self.crop_after_resize[1]))
            else:
                # crop from the center
                x = max(0, (tfunction.get_image_size(A)[0] - self.crop_after_resize[0]) // 2)
                y = max(0, (tfunction.get_image_size(A)[1] - self.crop_after_resize[1]) // 2)
            
            # crop A, B and N
            A = tfunction.crop(A, y, x, self.crop_after_resize[0], self.crop_after_resize[1])
            B = tfunction.crop(B, y, x, self.crop_after_resize[0], self.crop_after_resize[1])

        # apply random vertical flip
        if self.vh_flip and random.random() > 0.5:
            A = tfunction.hflip(A)
            B = tfunction.hflip(B)
        
        # clamp to 0.0,1.0 range
        A = torch.clamp(A, min=0.0, max=1.0)
        B = torch.clamp(B, min=0.0, max=1.0)
        
        return A, B
    
    # return the image size
    def GetImageSize(self):
        if self.crop_after_resize[0] >= 0:
            return self.crop_after_resize
        else:
            return self.resize_after_load