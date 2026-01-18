import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import median_filter
from collections import defaultdict
from utility.GetDevice import GetDevice
import torchvision.transforms.functional as tfunction

# assumes that images are labeled as 00000_00.png, 00000_01.png
class VideoDataLoader:
    def __init__(self, dataset_config, aggregation_size):
    
        # assert on error
        config_file = os.path.join("dataloader", "configs", f"{dataset_config}.json")
        assert os.path.isfile(config_file), f"config file not found at {config_file}"
        
        # set device
        self.device = GetDevice()
        
        # try to read configuration file and set variables
        with open(config_file, 'r') as file:
            config = json.load(file)
        
            # get image paths
            self.irn_paths = self.GetFilePaths(config["dataset_path"], config["subsets"])
            self.use_rgb = False

            # create keys list
            self.irn_keys = list(self.irn_paths.keys())

            # add the necessary configuration options
            self.name = config["dataset_name"]
            self.rgb_multiplier = torch.tensor(1.0 if self.use_rgb else 0.0, device=self.device)
            self.video_count = len(self.irn_keys)
            
            # log status
            print(f"{self.name} dataset scanned successfully with {self.video_count} unique videos")
            
        # dataset options
        self.aggregation_size = aggregation_size    # number of noisy image for each FPN pattern
        self.video_idx = 0
        self.image_idx = 0
        self.lower = 0
        self.higher = 65535
        self.device = GetDevice()

        # print info
        print("Number of images in the VIDEO set is {}".format(self.video_count))
    
    def __iter__(self):
        self.image_idx = 0
        self.video_idx = 0
        self.lower = 0
        self.higher = 65535
        return self
    
    # returns fpn [n x 2 x H x W], img [k x 1 x H x W]
    def __next__(self):
        if self.video_idx >= self.video_count:
            raise StopIteration

        # get rgb and irc img paths
        return_video_name = self.irn_keys[self.video_idx]
        return_video_frame = self.image_idx
        irn_selected = self.irn_paths[self.irn_keys[self.video_idx]]

        # compute 16->8 bit transformation at first
        if self.image_idx == 0:
            [self.lower, self.higher] = self.compute_video_percentiles(irn_selected)

        # convert numpy to tensor
        rgb_imgs = []
        irn_imgs = []
        for idx in range(self.aggregation_size):
            # get #aggregation_size image, loop if not available
            img_idx = (idx + self.image_idx) % len(irn_selected)
            irn_img = self.load_16bit_image(irn_selected[img_idx])
            rgb_img = median_filter(irn_img, size=(9, 19, 1)).repeat(3, 2)

            rgb = tfunction.to_tensor(rgb_img).to(self.device)
            irn = tfunction.to_tensor(irn_img).to(self.device)

            # append images
            rgb_imgs.append(rgb)
            irn_imgs.append(irn)

        # update the next index, if end go to next video
        self.image_idx = self.image_idx + 1
        if self.image_idx == len(irn_selected):
            self.video_idx = self.video_idx + 1
            self.image_idx = 0
        
        # return noisy ir and rgb image
        return rgb_imgs, irn_imgs, return_video_name, return_video_frame
        
    def __len__(self):
        return self.video_count

    # return the images inside the given directory
    @staticmethod
    def GetFilePaths(folder, subsets):
        grouped_filenames = defaultdict(list)
        for subset in subsets:
            subset_folder = os.path.join(folder, subset)
            for root, sequence, filenames in os.walk(subset_folder):
                filenames = sorted(filenames)
                for filename in filenames:
                    input_path = os.path.abspath(root)
                    file_path = os.path.join(input_path, filename)
                    if filename.endswith('.png') or filename.endswith('.jpg'):
                        parts = file_path.split(os.sep)
                        key = f"{subset}_{parts[-2]}"
                        grouped_filenames[key].append(file_path)
        return grouped_filenames

    # reads 16bit image
    def load_16bit_image(self, img_path):
        img16 = np.array(Image.open(img_path), dtype=np.uint16)
        img = img16.astype(np.float32)
        img = (img - self.lower) / (self.higher - self.lower + 1e-6)
        img = np.clip(img, 0.0, 1.0)
        return np.expand_dims((img * 255).astype(np.uint8),2)

    # get the percentile based histogram limits for all dataset
    def compute_video_percentiles(self, frame_paths, p_low=0.5, p_high=99.5):
        pixels = []
        for p in frame_paths:
            img = np.array(Image.open(p), dtype=np.uint16)
            pixels.append(img.reshape(-1))

        pixels = np.concatenate(pixels)
        lo = np.percentile(pixels, p_low)
        hi = np.percentile(pixels, p_high)
        return lo, hi

