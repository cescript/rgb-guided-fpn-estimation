import os
import torch
import models
import argparse
from itertools import product
from metrics import MetricEvaluator
from utility.OutputVisualizer import OutputVisualizer
from utility.FPNUtility import fpn_insert, fpn_remove
from models.safta.SaftaDenoiser import SAFTADenoiser

# import required classes
from dataloader.NoiseImageDataLoader import NoiseImageDataLoader

# get the test configuration
def get_test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_results', type=int, default=0)
    return parser.parse_args()

# run test code
if __name__ == '__main__':

    # get the test configuration
    opt = get_test_config()

    # setup ablation parameters and options
    base_model_name = "SAFTA-RGB-NIL"
    aggregation = 12
    dataset_names = ["m3fd_config"]
    fusion_models = ["basic", "attention", "multiscale"]
    fpn_types = ["fpn", "hfn"]
    shift_values = [0]

    # loop over all cases
    for dataset, shift, fpn_type in product(dataset_names, shift_values, fpn_types):

        # create test output name
        top_folder = os.path.join("output", "fusion_ablation")
        output_folder = f"{dataset}_K_{aggregation}_{fpn_type}_{shift}"

        # create a visualizer and metric evaluator
        visualize = OutputVisualizer(top_folder, output_folder, save_logs=True)
        metric_evaluator = MetricEvaluator(top_folder, output_folder)

        # loop over all models and test the results
        for mid, fusion_name in enumerate(fusion_models):
            print("Evaluating %s strategy..." % fusion_name)

            # create dataset loader and fpn generator for training
            fpn_img_loader = NoiseImageDataLoader(dataset, 1, aggregation, fpn_type, is_train_mode=False)

            # create a model given opt.model and other options
            model = models.GenerateModel(base_model_name)

            # change the denoiser of the model and load the weight
            model.denoiser = SAFTADenoiser(is_train_mode=False, fusion_block=fusion_name).to(model.device)
            model.denoiser.load_state(os.path.join("models", "safta", "weights", "ablation_weights", f"{fusion_name}_denoiser_20.pth"))

            # start the evaluation of the model
            model_name = f"{base_model_name}_{fusion_name}"
            metric_evaluator.start(model_name)

            # start the image loader
            # fpn_img: B x 2 x HEIGHT x WIDTH, rgb_img: K x 3 x HEIGHT x WIDTH, irc_img: K x 1 x HEIGHT x WIDTH
            image_index = 0
            for fpn_img, rgb_img, irc_img in fpn_img_loader:
                # irn_imgs: K x [1 x HEIGHT x WIDTH]
                # rgb_imgs: K x [3 x HEIGHT x WIDTH]
                irn_imgs = [fpn_insert(irc_img[idx], fpn_img).squeeze(0) for idx in range(irc_img.shape[0])]
                rgb_imgs = [rgb_img[idx] for idx in range(rgb_img.shape[0])]

                # apply fixed spatial shift to RGB images for misalignment robustness test
                if shift > 0:
                    rgb_imgs = [torch.roll(torch.roll(r, shift, dims=-1), shift, dims=-2) for r in rgb_imgs]

                # fpn_imgs: K x [2 x HEIGHT x WIDTH]
                fpn_imgs = model.forward(irn_imgs, rgb_imgs)

                # ire_imgs: K x [1 x HEIGHT x WIDTH]
                ire_imgs = [fpn_remove(irn_imgs[idx], fpn_imgs[idx].unsqueeze(0)).squeeze(0) for idx in range(len(fpn_imgs))]

                # save the first image
                if image_index <= opt.save_results:
                    visualize.save_fpn_results(model_name, image_index, fpn_img[0], fpn_imgs[0])
                    visualize.save_inference_results(model_name, image_index, rgb_imgs[0], irc_img[0], irn_imgs[0], ire_imgs[0])

                # evaluate the result of the clean image
                metric_evaluator.evaluate(irc_img, torch.stack(ire_imgs, dim=0))
                image_index += 1

            # save the metric to given file
            metric_evaluator.save_metrics(reduction="mean")
        
    
    