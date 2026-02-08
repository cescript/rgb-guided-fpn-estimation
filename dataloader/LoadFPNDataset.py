import math
import torch
import random
import numpy as np
import torchvision.transforms.functional as tfunction

# custom FPN dataset
class LoadFPNDataset:

    def __init__(self, image_size, noise_count, fpn_type, device):
    
        # get the copy of the options
        self.image_size = image_size
        self.noise_count = noise_count
        self.device = device
        
        # noise parameters
        self.alpha_scale = 0.015
        self.beta_sigma  = 0.015
        
        # narcissism parameters
        self.n_sigma = [0.1, 0.2]
        self.n_square = 32
        self.n_gain = [-0.1, 0.1]

        # pick narcissism probability
        if fpn_type == "fpn":
            self.n_probability = 1
        elif fpn_type == "hfn":
            self.n_probability = 0
        elif fpn_type == "mixed":
            self.n_probability = 0.5
        else:
            print(f"invalid fpn type, use fpn/hfn/mixed")
            exit(1)

        # create noise patterns
        self.noise_patterns = []
        for idx in range(self.noise_count):
            self.noise_patterns.append(self.GenerateNoise())

        print(f"FPN noise dataset loaded successfully with {self.noise_count} unique {fpn_type}")
        
    def __getitem__(self, index):
        # sample noise patterns from dataset
        FPN = [tfunction.to_tensor(self.noise_patterns[i % self.noise_count]) for i in index]
        return torch.stack(FPN).to(self.device)
    
    def __len__(self):
        return self.noise_count
    
    # generate FPN noise
    def GenerateNoise(self):
    
        # create pattern
        pattern = np.ndarray([self.image_size[0], self.image_size[1], 2], dtype=np.single)
        pattern[:, :, 0] = 1.0  # alpha
        pattern[:, :, 1] = 0.0  # beta

        # insert column noise
        for i in range(self.image_size[1]):
            pattern[:, i, 0] = random.uniform(1 - self.alpha_scale, 1 + self.alpha_scale)
            pattern[:, i, 1] = random.gauss(0, self.beta_sigma)
        
        # create narcissism
        if self.n_probability >= random.uniform(0, 1):
            cx = self.image_size[1] // 2 + random.randint(-self.n_square, self.n_square)
            cy = self.image_size[0] // 2 + random.randint(-self.n_square, self.n_square)
            s1 = math.sqrt(self.image_size[0]**2 + self.image_size[1]**2) * random.uniform(self.n_sigma[0], self.n_sigma[1])

            # update the beta term (gaussian output will be (0.0,0.1) range)
            gain = random.uniform(self.n_gain[0], self.n_gain[1])
            pattern[:, :, 1] += gain * self.gaus2d(self.image_size, cx, cy, s1, s1)
        
        # return the stripe and radial noise
        return pattern

    # define un-normalized 2D gaussian
    @staticmethod
    def gaus2d(crop_size, mx=0, my=0, sx=1, sy=1):
        x, y = np.indices((crop_size[1], crop_size[0]))
        return np.exp(-((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.))).transpose()
