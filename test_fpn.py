import os
import torch
import models
import argparse
import numpy as np
from itertools import product
from utility.FPNUtility import fpn_insert

# import required classes
from dataloader.NoiseImageDataLoader import NoiseImageDataLoader


# compute MAE and RMSE between predicted and ground truth FPN parameter maps
def compute_param_metrics(fpn_pred, fpn_gt):
    """Compare predicted and ground truth FPN parameters (alpha and beta).
    Args:
        fpn_pred: [2, H, W] predicted parameters (channel 0: alpha, channel 1: beta)
        fpn_gt:   [2, H, W] ground truth parameters (channel 0: alpha, channel 1: beta)
    Returns:
        dict with alpha_mae, alpha_rmse, beta_mae, beta_rmse
    """
    alpha_diff = fpn_pred[0].float() - fpn_gt[0].float()
    beta_diff = fpn_pred[1].float() - fpn_gt[1].float()

    return {
        "alpha_mae": alpha_diff.abs().mean().item(),
        "alpha_rmse": alpha_diff.pow(2).mean().sqrt().item(),
        "beta_mae": beta_diff.abs().mean().item(),
        "beta_rmse": beta_diff.pow(2).mean().sqrt().item(),
    }


# get the test configuration
def get_test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_results', type=int, default=0)
    return parser.parse_args()


# run test code
if __name__ == '__main__':

    # get the test configuration
    opt = get_test_config()

    # fpn parameter accuracy (similar to k_effect but only SAFTA-RGB)
    dataset_names = ["m3fd_config"]
    model_names = ["EMPTY", "SAFTA-RGB"]
    aggregation_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    fpn_types = ["fpn", "hfn"]

    # loop over all cases
    for dataset, aggregation, fpn_type in product(dataset_names, aggregation_sizes, fpn_types):

        # create test output name
        top_folder = os.path.join("output", "fpn_accuracy")
        output_folder = f"{dataset}_K_{aggregation}_{fpn_type}"
        save_dir = os.path.join(top_folder, output_folder)
        os.makedirs(save_dir, exist_ok=True)

        # loop over all models and test the results
        for mid, model_name in enumerate(model_names):
            print("Evaluating %s model..." % model_name)

            # create dataset loader and fpn generator for training
            fpn_img_loader = NoiseImageDataLoader(dataset, 1, aggregation, fpn_type, is_train_mode=False)

            # create a model given opt.model and other options
            model = models.GenerateModel(model_name)

            # accumulators for per-step metrics
            metrics_per_k = []

            # start the image loader
            # fpn_img: B x 2 x HEIGHT x WIDTH, rgb_img: K x 3 x HEIGHT x WIDTH, irc_img: K x 1 x HEIGHT x WIDTH
            image_index = 0
            for fpn_img, rgb_img, irc_img in fpn_img_loader:
                # irn_imgs: K x [1 x HEIGHT x WIDTH]
                # rgb_imgs: K x [3 x HEIGHT x WIDTH]
                irn_imgs = [fpn_insert(irc_img[idx], fpn_img).squeeze(0) for idx in range(irc_img.shape[0])]
                rgb_imgs = [rgb_img[idx] for idx in range(rgb_img.shape[0])]

                # fpn_imgs: K x [2 x HEIGHT x WIDTH]
                fpn_imgs = model.forward(irn_imgs, rgb_imgs)

                # evaluate parameter accuracy at each step k
                metrics_per_k.append(compute_param_metrics(fpn_imgs[0], fpn_img[0]))

                image_index += 1

            # aggregate per-step results
            k = aggregation
            a_mae = np.mean([m["alpha_mae"] for m in metrics_per_k])
            a_rmse = np.mean([m["alpha_rmse"] for m in metrics_per_k])
            b_mae = np.mean([m["beta_mae"] for m in metrics_per_k])
            b_rmse = np.mean([m["beta_rmse"] for m in metrics_per_k])

            # print per-step results
            print(f"\n  {model_name} | {dataset} | {fpn_type} | K={aggregation}")
            print(f"  {'k':>3} | {'a_MAE':>10} | {'a_RMSE':>10} | {'b_MAE':>10} | {'b_RMSE':>10}")
            print(f"  {k:>3} | {a_mae:>10.6f} | {a_rmse:>10.6f} | {b_mae:>10.6f} | {b_rmse:>10.6f}")
            print()

            # save per-step results to csv
            csv_path = os.path.join(save_dir, f"{model_name}_fpn_metrics.csv")
            with open(csv_path, "w") as f:
                f.write("k,alpha_mae,alpha_rmse,beta_mae,beta_rmse\n")
                f.write(f"{k},{a_mae:.6f},{a_rmse:.6f},{b_mae:.6f},{b_rmse:.6f}\n")
            print(f"  Results saved to: {csv_path}")