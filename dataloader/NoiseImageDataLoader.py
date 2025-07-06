import torch
import random
from .LoadImageDataset import LoadImageDataset
from .LoadFPNDataset import LoadFPNDataset

class NoiseImageDataLoader:
    def __init__(self, dataset_config: str, batch_size:int, aggregation_size: int, is_train_mode: bool):
    
        # dataset options
        self.noise_count = 10000                    # total noise count
        self.random_seed = 12237505                 # random seed for the tests
        self.train_test_ratio_img = [0.7, 0.3]      # train and test ratio for the image dataset
        self.train_test_ratio_fpn = [0.9, 0.1]      # train and test ratio for the fpn dataset
        self.aggregation_size = aggregation_size    # number of noisy image for each FPN pattern
        self.current_idx = 0                        # set the current to zero
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # set the batch size and vertical/horizontal flip
        if is_train_mode:
            self.batch_size = batch_size
            self.vh_flip = True
            self.crop_random = True
        else:
            self.batch_size = batch_size
            self.vh_flip = False
            self.crop_random = False

        # set the random seed for consistency
        random.seed(self.random_seed)
        
        # dataset should have __init__, __get_item__ and __len__ methods
        self.image_dataset = LoadImageDataset(dataset_config, self.vh_flip, self.crop_random, self.device)
        self.fpn_dataset   = LoadFPNDataset(self.image_dataset.GetImageSize(), self.noise_count, self.device)
        
        # create shuffled image indices
        image_indices = list(range(len(self.image_dataset)))
        random.shuffle(image_indices)

        # create shuffled noise indices
        noise_indices = list(range(len(self.fpn_dataset)))
        random.shuffle(noise_indices)

        # get indices
        if is_train_mode:
            image_index = round(len(self.image_dataset) * self.train_test_ratio_img[0])
            noise_index = round(len(self.fpn_dataset) * self.train_test_ratio_fpn[0])
            self.image_indices = image_indices[:image_index]
            self.noise_indices = noise_indices[:noise_index]
        else:
            image_index = round(len(self.image_dataset) * self.train_test_ratio_img[1])
            noise_index = round(len(self.fpn_dataset) * self.train_test_ratio_fpn[1])
            self.image_indices = image_indices[-image_index:]
            self.noise_indices = noise_indices[-noise_index:]
        
        # print info
        print("Number of images in the {} set is {}".format("TRAIN" if is_train_mode else "TEST", len(self.image_indices)))
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    # returns fpn [n x 2 x H x W], img [k x 1 x H x W]
    def __next__(self):
        if self.current_idx >= len(self.noise_indices):
            raise StopIteration
        
        # get the next batch of noise
        next_idx = min(self.current_idx + self.batch_size, len(self.noise_indices))
        batch_noise_indices = self.noise_indices[self.current_idx : next_idx]
        fpn_img = self.fpn_dataset[batch_noise_indices]
        
        # get a random subset of image
        batch_image_indices = random.sample(self.image_indices, self.aggregation_size)
        rgb_img, irc_img = self.image_dataset[batch_image_indices]

        # update the next index
        self.current_idx = next_idx
        
        # return fpn, {rgb, clean ir images}
        return fpn_img, rgb_img, irc_img
        
    def __len__(self):
        return len(self.noise_indices) // self.batch_size
    
    def get_image_size(self):
        return self.image_dataset.GetImageSize()
