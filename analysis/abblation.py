import os
import torch
import models
import argparse
from itertools import product
from metrics import MetricEvaluator
from utility.OutputVisualizer import OutputVisualizer
from utility.FPNUtility import fpn_insert, fpn_remove

# import required classes
from dataloader.NoiseImageDataLoader import NoiseImageDataLoader


# get the test configuration
def get_test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--aggregation_size', type=int, default=8)
    parser.add_argument('--dataset', type=str, default="m3fd_config")
    return parser.parse_args()

def get_inference_results(test_case, image_index, fpn_img, rgb_img, irc_img, rgb_weight, irn_weight, fpn_estimator):
    # irn_imgs: K x [1 x HEIGHT x WIDTH]
    # rgb_imgs: K x [3 x HEIGHT x WIDTH]
    irn_imgs = [fpn_insert(irc_img[idx], fpn_img).squeeze(0) for idx in range(irc_img.shape[0])]
    rgb_imgs = [rgb_img[idx] for idx in range(rgb_img.shape[0])]
    
    # fpn_imgs: K x [2 x HEIGHT x WIDTH]
    irn_weighted_images = [irn * irn_weight for irn in irn_imgs]
    rgb_weighted_images = [rgb * rgb_weight for rgb in rgb_imgs]
    
    # use fpn estimator to get ire_img
    if fpn_estimator:
        # use weighted image list for inference
        fpn_imgs = model.forward(irn_weighted_images, rgb_weighted_images)
        
        # ire_imgs: K x [1 x HEIGHT x WIDTH]
        ire_imgs = [fpn_remove(irn_imgs[idx], fpn_imgs[idx].unsqueeze(0)).squeeze(0) for idx in range(len(fpn_imgs))]
        
    # get ire from the denoiser
    else:
        ire_tensor = model.denoiser.forward(torch.stack(rgb_weighted_images, dim=0), torch.stack(irn_weighted_images, dim=0))
        ire_imgs = list(torch.unbind(ire_tensor, dim=0))

    # save the first image
    if image_index == 0:
        visualize.save_inference_results(test_case, image_index, rgb_imgs[0], irc_img[0], irn_imgs[0], ire_imgs[0])
        
    # return the ire estimate
    return ire_imgs

# run test code
if __name__ == '__main__':
    # get the test configuration
    opt = get_test_config()
    
    # create a visualizer
    visualize = OutputVisualizer("output", "ablation_study", save_logs=True)

    # set the model names to be evaluated and test scenario
    model_name = "SAFTA-RGB"
    rgb_weights = [0.0, 1.0]
    irn_weights = [1.0]
    fpn_estimators = [True, False]
    
    # create a model given opt.model and other options
    model = models.GenerateModel(model_name)

    # create a metric evaluator class
    metric_evaluator = MetricEvaluator("output", "ablation_study")
    
    # get all possible weight combinations
    for rgb_w, irn_w, fpn_flag in product(rgb_weights, irn_weights, fpn_estimators):
        test_case = f"rgb{int(rgb_w)}_irn{int(irn_w)}_fpn{int(fpn_flag)}"
        print("Evaluating %s test..." % test_case)

        # create dataset loader and fpn generator for training
        fpn_img_loader = NoiseImageDataLoader(opt.dataset, opt.batch_size, opt.aggregation_size, is_train_mode=False)

        # start the evaluation of the model
        metric_evaluator.start(test_case)

        # create a model given opt.model and other options
        model = models.GenerateModel(model_name)

        # start the image loader
        image_index = 0
        for fpn_img, rgb_img, irc_img in fpn_img_loader:
            # get the result
            ire_imgs = get_inference_results(test_case, image_index, fpn_img, rgb_img, irc_img, rgb_w, irn_w, fpn_flag)
    
            # evaluate the result of the clean image
            metric_evaluator.evaluate(irc_img, torch.clamp(torch.stack(ire_imgs, dim=0), 0.0, 1.0))
            image_index += 1

        # save the metric to given file
        metric_evaluator.save_metrics(reduction="mean")