import torch
import os
import time
from functools import partial
from piq import psnr, ssim, gmsd, brisque, CLIPIQA
import numpy as np

class MetricEvaluator:
    def __init__(self, path, subfolder, reference=True):
        """ create a metric evaluator for IRFPN denoising """
        output_directory = os.path.join(path, subfolder)
        self.filename = os.path.join(output_directory, "scores.txt")

        if reference is True:
            self.metrics = "psnr, ssim, gmsd"
            self.metricFun = [
                partial(psnr, reduction='none'),
                partial(ssim, reduction='none'),
                partial(gmsd, reduction='none')
            ]
        else:
            self.metrics = "brisque, clip_iqa"
            clip_iqa = CLIPIQA()
            self.metricFun = [
                partial(brisque, reduction='none'),
                partial(clip_iqa)
            ]

        # create output directory for the scores
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # check the score file and delete if exist
        if os.path.isfile(self.filename):
            os.remove(self.filename)
            
    # set the current header
    def start(self, modelName):
        self.header = modelName + "," + self.metrics
        self.scores = np.zeros((0, len(self.metricFun)))
        self.start_time = time.perf_counter()
    
    # assumes that ref and irc has Nx1xHxW
    def evaluate(self, ref, irc):
        """" calculate the statistics of each element """
        # get the scores
        mcount = len(self.metricFun)
        score = np.zeros((irc.shape[0], mcount))

        # decide which method to use and analyze result
        if ref is not None:
            for m in range(mcount):
                score[:, m] = self.metricFun[m](ref.cpu(), irc.cpu())
        else:
            for m in range(mcount):
                val = self.metricFun[m](irc.cpu())
                score[:, m] = val.view(val.shape[0])
        
        # append the array
        self.scores = np.append(self.scores, score, axis=0)

    # save the resulting scores array to disk
    def save_metrics(self, reduction: str):
        # calculate avg time per image
        elapsed = time.perf_counter() - self.start_time
        samples = self.scores.shape[0]
        avg_time = (elapsed / samples) * 1000 # ms

        # append avgtime column to all rows
        avg_col = np.full((samples, 1), avg_time)
        scores_with_time = np.concatenate([self.scores, avg_col], axis=1)
        header_with_time = self.header + "," + "avgtime"

        # open the so that we can append multiple results
        with open(self.filename, "a") as log_file:
            # print each score
            if reduction == "none":
                np.savetxt(log_file, scores_with_time, fmt='%.4f', delimiter=', ', header=header_with_time)
            elif reduction == "mean":
                np.savetxt(log_file, np.mean(scores_with_time, axis=0, keepdims=True), fmt='%.4f', delimiter=', ', header=header_with_time)
            else:
                print("unknown reduction method")
        
        
        