import os
import time
import argparse
from utility.FPNUtility import fpn_insert

# import required classes
from models.safta.SaftaDenoiser import SAFTADenoiser
from utility.OutputVisualizer import OutputVisualizer
from dataloader.NoiseImageDataLoader import NoiseImageDataLoader

# get the training configuration
def get_training_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--aggregation_size', type=int, default=12)
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
    fpn_img_loader = NoiseImageDataLoader(opt.dataset, opt.batch_size, opt.aggregation_size, "mixed", is_train_mode=True)
    
    # create model and load parameters if exist
    total_epochs = opt.n_epochs_decay + opt.n_epochs
    safta_model = SAFTADenoiser(opt.learning_rate, opt.n_epochs_decay, total_epochs, is_train_mode=True)
    
    # create a visualizer
    top_folder = os.path.join("output", "train")
    visualize = OutputVisualizer(top_folder, "denoiser", save_logs=True)
    
    # load the pretrained weights if exist
    weights_directory = os.path.join(top_folder, "denoiser")
    if opt.epoch_count > 1:
        print(f"Loading previously trained model")
        safta_model.load_state(os.path.join(weights_directory, f"denoiser_{opt.epoch_count}.pth"))

    # create output directory
    os.makedirs(top_folder, exist_ok=True)

    # outer loop for different epochs
    for epoch in range(opt.epoch_count, total_epochs + 1):
        # timer for entire epoch
        epoch_start_time = time.time()
        
        # get a set of data -> fpn [n x 2 x H x W], rgb_img/irc_img [k x 1 x H x W]
        total_loss = 0
        total_images = 0
        for fpn_img, rgb_img, irc_img in fpn_img_loader:
            # each batch contains [k x 1 x H x W] images
            for idx in range(rgb_img.shape[0]):
                # apply fpn to clean image -> [n x 1 x H x W]
                irn_img = fpn_insert(irc_img[idx], fpn_img)
                
                # iterate over the kth images
                rgb_img_input = rgb_img[idx].unsqueeze(0).repeat(irn_img.shape[0], 1, 1, 1)
                irc_est = safta_model.forward(rgb_img_input, irn_img)
                
                # optimize parameters for the current batch
                total_images += opt.batch_size
                # repeat target to match batch size
                irc_img_rep = irc_img[idx].unsqueeze(0).repeat(irc_est.shape[0], 1, 1, 1)
                total_loss += safta_model.optimize_parameters(irc_img_rep, irc_est)
                
                # save an example result at the end of the epoch
                if total_images == opt.batch_size * opt.aggregation_size * len(fpn_img_loader):
                    visualize.save_denoising_results(epoch, rgb_img[idx], irc_img[idx], irn_img[0], irc_est[0])

        # now update learning rate
        safta_model.step_scheduler()
        safta_model.save_state(os.path.join(weights_directory, f"denoiser_{epoch}.pth"))

        # good place to print some insight
        duration = time.time() - epoch_start_time
        visualize.print_current_losses(epoch, total_epochs, duration, {'epoch_loss': total_loss / total_images})
