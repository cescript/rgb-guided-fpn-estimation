import torch
import models
import argparse
from metrics import MetricEvaluator
from utility.OutputVisualizer import OutputVisualizer
from utility.FPNUtility import fpn_insert, fpn_remove

# import required classes
from dataloader.RealImageDataLoader import RealImageDataLoader

# get the test configuration
def get_test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--aggregation_size', type=int, default=8)
    parser.add_argument('--dataset', type=str, default="tendero")
    return parser.parse_args()

# run test code
if __name__ == '__main__':
    # get the test configuration
    opt = get_test_config()

    # create a visualizer
    visualize = OutputVisualizer("output", "tendero", save_logs=True)
    
    # set the model names to be evaluated
    model_names = ["EMPTY", "SAFTA-RGB", "D1WLS", "DLSNUC", "MULTIVIEW"]

    # create a metric evaluator class
    metric_evaluator = MetricEvaluator("output", "tendero")
    
    # loop over all models and test the results
    for mid, model_name in enumerate(model_names):
        print("Evaluating %s model..." % model_name)

        # create a real world dataset loader
        noisy_img_loader = RealImageDataLoader(opt.dataset, opt.aggregation_size)

        # start the evaluation of the model
        metric_evaluator.start(model_name)

        # create a model given opt.model and other options
        model = models.GenerateModel(model_name)
        
        # start the image loader
        # irn_imgs: K x [1 x HEIGHT x WIDTH]
        # rgb_imgs: K x [3 x HEIGHT x WIDTH]
        image_index = 0
        for rgb_imgs, irn_imgs in noisy_img_loader:

            # fpn_imgs: K x [2 x HEIGHT x WIDTH]
            fpn_imgs = model.forward(irn_imgs, rgb_imgs)

            # ire_imgs: K x [1 x HEIGHT x WIDTH]
            ire_imgs = [fpn_remove(irn_imgs[idx], fpn_imgs[idx].unsqueeze(0)).squeeze(0) for idx in range(len(fpn_imgs))]
            
            # save the first image
            visualize.save_fpn_results(model_name, image_index, fpn_imgs[0], fpn_imgs[0])
            visualize.save_inference_results(model_name, image_index, rgb_imgs[0], irn_imgs[0], irn_imgs[0], ire_imgs[0])
            
            # evaluate the result of the clean image
            image_index += 1
            
        
    
    