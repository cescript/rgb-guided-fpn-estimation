import os
import json
import torch
from PIL import Image
from collections import defaultdict
from utility.GetDevice import GetDevice
import torchvision.transforms.functional as tfunction

# assumes that images are labeled as 00000_00.png, 00000_01.png
class RealImageDataLoader:
    def __init__(self, dataset_config, aggregation_size):
    
        # assert on error
        config_file = os.path.join("dataloader", "dataset", f"{dataset_config}.json")
        assert os.path.isfile(config_file), f"config file not found at {config_file}"
        
        # set device
        self.device = GetDevice()
        
        # try to read configuration file and set variables
        with open(config_file, 'r') as file:
            config = json.load(file)
        
            # get image paths
            dataset_path = config["dataset_path"]
            self.rgb_paths = self.GetFilePaths(os.path.join(dataset_path, config["rgb_subdirectory"]))
            self.irn_paths = self.GetFilePaths(os.path.join(dataset_path, config["irc_subdirectory"]))
            
            # check valid rgb pair provided
            self.use_rgb = True
            if len(self.rgb_paths) == 0:
                # set rgb paths to valid path to prevent errors
                # RGB images will be ZEROed after reading it
                self.rgb_paths = self.irn_paths
                self.use_rgb = False
            
            # create keys list
            self.rgb_keys = list(self.rgb_paths.keys())
            self.irn_keys = list(self.irn_paths.keys())
            
            # assert if the size doesnt match
            assert len(self.rgb_keys) == len(self.irn_keys), "rgb and ir images must have the same number of images"

            # add the necessary configuration options
            self.name = config["dataset_name"]
            
            self.resize_after_load = config["resize_after_load"]
            self.rgb_multiplier = torch.tensor(1.0 if self.use_rgb else 0.0, device=self.device)
            self.image_count = len(self.irn_keys)
            
            # log status
            print(f"{self.name} dataset scanned successfully with {self.image_count} unique images")
            
        # dataset options
        self.aggregation_size = aggregation_size    # number of noisy image for each FPN pattern
        self.current_idx = 0                        # set the current to zero
        self.device = GetDevice()

        # print info
        print("Number of images in the TEST set is {}".format(self.image_count))
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    # returns fpn [n x 2 x H x W], img [k x 1 x H x W]
    def __next__(self):
        if self.current_idx >= self.image_count:
            raise StopIteration
        
        # get rgb and irc img paths
        rgb_selected = self.rgb_paths[self.rgb_keys[self.current_idx]]
        irn_selected = self.irn_paths[self.irn_keys[self.current_idx]]

        # convert numpy to tensor
        rgb_imgs = []
        irn_imgs = []
        for idx in range(self.aggregation_size):
            # get #aggregation_size image, loop if not available
            rgb_img = Image.open(rgb_selected[idx % len(rgb_selected)]).convert('RGB')
            irn_img = Image.open(irn_selected[idx % len(irn_selected)]).convert('L')

            rgb = tfunction.to_tensor(rgb_img).to(self.device)
            irn = tfunction.to_tensor(irn_img).to(self.device)

            # resize after load
            rgb = tfunction.resize(rgb, self.resize_after_load, interpolation=tfunction.InterpolationMode.BILINEAR)
            irn = tfunction.resize(irn, self.resize_after_load, interpolation=tfunction.InterpolationMode.BILINEAR)
            
            # append images
            rgb_imgs.append(rgb * self.rgb_multiplier)
            irn_imgs.append(irn)
            
        # update the next index
        self.current_idx = self.current_idx + 1
        
        # return noisy ir and rgb image
        return rgb_imgs, irn_imgs
        
    def __len__(self):
        return self.image_count
    
    def get_image_size(self):
        return self.image_count

    # return the images inside the given directory
    @staticmethod
    def GetFilePaths(folder):
        grouped_filenames = defaultdict(list)
        for root, dirs, filenames in os.walk(folder):
            filenames = sorted(filenames)
            for filename in filenames:
                input_path = os.path.abspath(root)
                file_path = os.path.join(input_path, filename)
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    key = filename.split('_')[0]
                    grouped_filenames[key].append(file_path)

            break  # prevent descending into subfolders
        return grouped_filenames
