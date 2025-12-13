import torch
import os
import time
from functools import partial
from piq import psnr, ssim, gmsd
import numpy as np

class MetricEvaluator:
    def __init__(self, path, subfolder):
        """ create a metric evaluator for IRFPN denoising """
        output_directory = os.path.join(path, subfolder)
        self.filename = os.path.join(output_directory, "scores.txt")
        self.metrics = "psnr, ssim, gmsd"
        self.metricFun = [
            partial(psnr, reduction='none'),
            partial(ssim, reduction='none'),
            partial(gmsd, reduction='none')
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
        score = np.zeros((ref.shape[0], mcount))
        for m in range(mcount):
            score[:, m] = self.metricFun[m](ref.cpu(), irc.cpu())
        
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
        header_with_time = self.header.append("avgtime")

        # open the so that we can append multiple results
        with open(self.filename, "a") as log_file:
            # print each score
            if reduction == "none":
                np.savetxt(log_file, scores_with_time, fmt='%.4f', delimiter=', ', header=header_with_time)
            elif reduction == "mean":
                np.savetxt(log_file, np.mean(scores_with_time, axis=0, keepdims=True), fmt='%.4f', delimiter=', ', header=header_with_time)
            else:
                print("unknown reduction method")
        
        
        