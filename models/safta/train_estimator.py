import os
import time
import math
import random
import argparse
from utility.FPNUtility import fpn_insert

# import required classes
from models.safta.SaftaNoiseEstimator import SaftaNoiseEstimator
from utility.OutputVisualizer import OutputVisualizer
from dataloader.NoiseImageDataLoader import NoiseImageDataLoader

# get the training configuration
def get_training_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--aggregation_size', type=int, default=8)
    parser.add_argument('--epoch_count', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=40)
    parser.add_argument('--n_epochs_decay', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--dataset', type=str, default="m3fd_config")
    return parser.parse_args()

# run training code
if __name__ == '__main__':
    
    # get the training configuration
    opt = get_training_config()

    # create dataset loader and fpn generator for training
    fpn_img_loader = NoiseImageDataLoader(opt.dataset, opt.batch_size, opt.aggregation_size, is_train_mode=True)

    # create model and load parameters if exist
    total_epochs = opt.n_epochs_decay + opt.n_epochs
    safta_noise_estimator = SaftaNoiseEstimator(opt.learning_rate, opt.n_epochs_decay, total_epochs, is_train_mode=True)

    # create a visualizer
    visualize = OutputVisualizer("output", "fpn_estimator", save_logs=True)
    
    # load the pretrained weights if exist
    weights_directory = os.path.join("output", "fpn_estimator")
    if opt.epoch_count > 1:
        print(f"loading previously trained model")
        safta_noise_estimator.load_state(os.path.join(weights_directory, f"epoch_{opt.epoch_count}.pth"))

    # create output directory
    os.makedirs("output", exist_ok=True)
    
    # outer loop for different epochs
    for epoch in range(opt.epoch_count, total_epochs + 1):
        # timer for entire epoch
        epoch_start_time = time.time()
        
        # get a set of data -> fpn [batch x 2 x H x W], img [agg x 1 x H x W]
        total_loss = 0
        total_images = 0
        for fpn_img, rgb_img, irc_img in fpn_img_loader:
            # reset SAFTA and iterate for all images
            safta_noise_estimator.reset_embedding()
            
            # pick one of the images as anchor and get clean and noisy versions of it
            anchor_idx = random.randint(0, rgb_img.shape[0] - 1)
            irn_anchor = fpn_insert(irc_img[anchor_idx], fpn_img)
            irc_anchor = irc_img[anchor_idx].unsqueeze(0).repeat(irn_anchor.shape[0],1,1,1)
            
            # each batch contains [agg x 1 x H x W] images
            for idx in range(rgb_img.shape[0]):
                # apply fpn to clean image -> [batch x 1 x H x W]
                irn_img = fpn_insert(irc_img[idx], fpn_img)
                rgb_img_input = rgb_img[idx].unsqueeze(0).repeat(irn_img.shape[0], 1, 1, 1)
                
                # predict the IRC image
                ire_image = irc_img[idx].unsqueeze(0).repeat(irn_img.shape[0], 1, 1, 1)

                # iterate over the kth image and get fpn and irc estimate
                fpn_est = safta_noise_estimator.forward(ire_image, irn_img)
                
                # accumulate the loss over anchor images
                iter_weight = math.sqrt(float(idx) / opt.aggregation_size)
                ire_anchor = safta_noise_estimator.accumulate_loss(irc_anchor, irn_anchor, fpn_est, iter_weight)
                
                # increase total images
                total_images += opt.batch_size

                # save an example result at the end of the epoch
                if total_images == opt.batch_size * opt.aggregation_size * len(fpn_img_loader):
                    visualize.save_fpn_results("safta", epoch, fpn_img[0], fpn_est[0])
                    visualize.save_denoising_results(epoch, rgb_img[anchor_idx], irc_img[anchor_idx], irn_anchor[0], ire_anchor[0])
                    
            # get gradients and update network weights
            total_loss += safta_noise_estimator.optimize_parameters()

        # now update learning rate
        safta_noise_estimator.step_scheduler()
        safta_noise_estimator.save_state(os.path.join(weights_directory, f"fpn_estimator_{epoch}.pth"))

        # good place to print some insight
        duration = time.time() - epoch_start_time
        visualize.print_current_losses(epoch, total_epochs, duration, {'epoch_loss': total_loss / total_images})