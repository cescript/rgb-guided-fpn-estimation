import torch
import models
import argparse
from metrics import MetricEvaluator
from utility.OutputVisualizer import OutputVisualizer
from utility.FPNUtility import fpn_insert, fpn_remove

# import required classes
from dataloader.NoiseImageDataLoader import NoiseImageDataLoader

# get the test configuration
def get_test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="m3fd_config")
    parser.add_argument('--save_results', type=int, default=50)
    return parser.parse_args()

# run test code
if __name__ == '__main__':
    # get the test configuration
    opt = get_test_config()

    # create a visualizer
    visualize = OutputVisualizer("output", "aggregation_comparison", save_logs=True)

    # set the model names to be evaluated
    aggregation_sizes = [2, 4, 6, 8, 10, 12]
    model_names = ["MULTIVIEW", "SAFTA-RGB", "SAFTA-RGB-OLS"]
    
    # create a metric evaluator class
    metric_evaluator = MetricEvaluator("output", "aggregation_comparison")
    
    # loop over all models for all aggregation size
    for _, aggregation_size in enumerate(aggregation_sizes):
        for mid, model_name in enumerate(model_names):
            print(f"Evaluating {model_name} for K={aggregation_size}...")
    
            # create dataset loader and fpn generator for training
            fpn_img_loader = NoiseImageDataLoader(opt.dataset, opt.batch_size, aggregation_size, is_train_mode=False)
    
            # start the evaluation of the model
            test_case_name = f"{model_name}-{aggregation_size}"
            metric_evaluator.start(test_case_name)
    
            # create a model given opt.model and other options
            model = models.GenerateModel(model_name)
            
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
    
                # ire_imgs: K x [1 x HEIGHT x WIDTH]
                ire_imgs = [fpn_remove(irn_imgs[idx], fpn_imgs[idx].unsqueeze(0)).squeeze(0) for idx in range(len(fpn_imgs))]
                
                # save the first image
                if image_index <= opt.save_results:
                    visualize.save_fpn_results(test_case_name, image_index, fpn_img[0], fpn_imgs[0])
                    visualize.save_inference_results(test_case_name, image_index, rgb_imgs[0], irc_img[0], irn_imgs[0], ire_imgs[0])
                
                # evaluate the result of the clean image
                metric_evaluator.evaluate(irc_img, torch.stack(ire_imgs, dim=0))
                image_index += 1
                
            # save the metric to given file
            metric_evaluator.save_metrics(reduction="mean")
        
    
    