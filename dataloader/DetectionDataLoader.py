import torch
import random
from utility.GetDevice import GetDevice
from .LoadImageDataset import LoadImageDataset
from .LoadFPNDataset import LoadFPNDataset


class DetectionDataLoader:
    def __init__(self, dataset_config: str, aggregation_size: int, fpn_type: str):
        # dataset options
        self.noise_count = 10000  # total noise count
        self.random_seed = 12237505  # random seed for the tests
        self.train_test_ratio_fpn = [0.9, 0.1]  # train and test ratio for the fpn dataset
        self.aggregation_size = aggregation_size  # number of noisy image for each FPN pattern
        self.current_idx = 0  # set the current to zero
        self.device = GetDevice()

        # no augmentation for detection dataset generation
        self.vh_flip = False
        self.crop_random = False

        # set the random seed for consistency
        random.seed(self.random_seed)

        # dataset should have __init__, __get_item__ and __len__ methods
        self.image_dataset = LoadImageDataset(dataset_config, self.vh_flip, self.crop_random, self.device)
        self.fpn_dataset = LoadFPNDataset(self.image_dataset.GetImageSize(), self.noise_count, fpn_type, self.device)

        # get the split ratio from config
        self.train_test_ratio_img = self.image_dataset.GetTrainTestRatio()

        # create shuffled image indices
        image_indices = list(range(len(self.image_dataset)))
        random.shuffle(image_indices)

        # create shuffled noise indices
        noise_indices = list(range(len(self.fpn_dataset)))
        random.shuffle(noise_indices)

        # split image indices into train and test
        image_split = round(len(self.image_dataset) * self.train_test_ratio_img[0])
        self.train_image_indices = image_indices[:image_split]
        self.test_image_indices = image_indices[-round(len(self.image_dataset) * self.train_test_ratio_img[1]):]

        # split noise indices into train and test
        noise_split = round(len(self.fpn_dataset) * self.train_test_ratio_fpn[0])
        self.test_noise_indices = noise_indices[-round(len(self.fpn_dataset) * self.train_test_ratio_fpn[1]):]

        # print info
        print("Number of test images: {}, support images: {}, noise patterns: {}".format(
            len(self.test_image_indices), len(self.train_image_indices), len(self.test_noise_indices)))

    def __iter__(self):
        self.current_idx = 0
        return self

    # returns fpn [1 x 2 x H x W], rgb_imgs [K x 3 x H x W], irc_imgs [K x 1 x H x W], target_index (always 0)
    def __next__(self):
        if self.current_idx >= len(self.test_image_indices):
            raise StopIteration

        # get the target test image index
        target_idx = self.test_image_indices[self.current_idx]

        # pick a noise pattern for this test image (cycle through test noise)
        noise_idx = self.test_noise_indices[self.current_idx % len(self.test_noise_indices)]
        fpn_img = self.fpn_dataset[[noise_idx]]

        # select K-1 support images from train split
        support_indices = random.sample(self.train_image_indices, self.aggregation_size - 1)

        # build the full K-image batch: target at index 0, support images after
        batch_indices = [target_idx] + support_indices
        rgb_img, irc_img = self.image_dataset[batch_indices]

        # update the next index
        self.current_idx += 1

        # return fpn, rgb, clean ir images (target is always at index 0)
        return fpn_img, rgb_img, irc_img

    def __len__(self):
        return len(self.test_image_indices)

    def get_image_size(self):
        return self.image_dataset.GetImageSize()

    def get_image_stem(self, test_idx):
        # get the original filename stem for the given test iteration index
        img_idx = self.test_image_indices[test_idx]
        return self.image_dataset.GetImageStem(img_idx)