import os
import torch
import models
import argparse
from metrics import MetricEvaluator
from utility.OutputVisualizer import OutputVisualizer
from utility.FPNUtility import fpn_insert, fpn_remove

# import required classes
from dataloader.VideoDataLoader import VideoDataLoader


# get the test configuration
def get_test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--aggregation_size', type=int, default=12)
    parser.add_argument('--dataset', type=str, default="butiv")
    parser.add_argument('--keep_fpn', type=int, default=1)
    return parser.parse_args()


# run test code
if __name__ == '__main__':
    # get the test configuration
    opt = get_test_config()

    # create a visualizer
    top_folder = os.path.join("output", "video_comparison")
    output_folder = f"{opt.dataset}_{opt.keep_fpn}"
    visualize = OutputVisualizer(top_folder, output_folder, save_logs=True)
    metric_evaluator = MetricEvaluator(top_folder, output_folder, reference=False)

    # set the model names to be evaluated
    model_names = ["SAFTA-RGB", "DCGAN", "MULTIVIEW", "D1WLS", "DLSNUC", "EMPTY"]

    # loop over all models and test the results
    for mid, model_name in enumerate(model_names):
        print("Evaluating %s model..." % model_name)

        # create a real world dataset loader
        noisy_img_loader = VideoDataLoader(opt.dataset, opt.aggregation_size, 10)

        # create a model given opt.model and other options
        model = models.GenerateModel(model_name)

        # start the evaluation of the model
        metric_evaluator.start(model_name)

        # start the image loader
        # irn_imgs: K x [1 x HEIGHT x WIDTH]
        # rgb_imgs: K x [3 x HEIGHT x WIDTH]
        for rgb_imgs, irn_imgs, video_name, frame_counter in noisy_img_loader:
            # fpn_imgs: K x [2 x HEIGHT x WIDTH]
            if frame_counter % opt.keep_fpn == 0:
                fpn_imgs = model.forward(irn_imgs, rgb_imgs)

            # ire_imgs: K x [1 x HEIGHT x WIDTH]
            ire_imgs = [fpn_remove(irn_imgs[idx], fpn_imgs[idx].unsqueeze(0)).squeeze(0) for idx in range(len(fpn_imgs))]

            # save the first image
            visualize.save_inference_results(video_name, frame_counter, rgb_imgs[0], irn_imgs[0], irn_imgs[0], ire_imgs[0])

            # evaluate the results
            metric_evaluator.evaluate(None, torch.stack(ire_imgs, dim=0))

        # save the metric to given file
        metric_evaluator.save_metrics(reduction="mean")



