import torch
import models
import argparse
from metrics import MetricEvaluator
from utility.OutputVisualizer import OutputVisualizer
from utility.FPNUtility import fpn_insert, fpn_remove
from utility.FPNBuffer import FPNBuffer

# import required classes
from dataloader.RealImageDataLoader import RealImageDataLoader

# get the test configuration
def get_test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--aggregation_size', type=int, default=12)
    parser.add_argument('--dataset', type=str, default="butiv_200")
    parser.add_argument('--save_fpn', type=bool, default=False)
    parser.add_argument('--keep_fpn', type=bool, default=False)
    return parser.parse_args()

# run test code
if __name__ == '__main__':
    # get the test configuration
    opt = get_test_config()

    # create a visualizer
    key_name = opt.dataset if opt.keep_fpn == False else f"{opt.dataset}_keep"
    visualize = OutputVisualizer("output", key_name, save_logs=True)
    
    # set the model names to be evaluated
    model_names = ["SAFTA-RGB", "DCGAN", "MULTIVIEW", "D1WLS", "DLSNUC", "EMPTY"]

    # create a metric evaluator class
    metric_evaluator = MetricEvaluator("output", key_name, reference=False)
    fpn_buffer = FPNBuffer("output", key_name)

    # loop over all models and test the results
    for mid, model_name in enumerate(model_names):
        print("Evaluating %s model..." % model_name)

        # create a real world dataset loader
        noisy_img_loader = RealImageDataLoader(opt.dataset, opt.aggregation_size)

        # create a model given opt.model and other options
        model = models.GenerateModel(model_name)

        # start the evaluation of the model
        metric_evaluator.start(model_name)
        fpn_buffer.start()

        # start the image loader
        # irn_imgs: K x [1 x HEIGHT x WIDTH]
        # rgb_imgs: K x [3 x HEIGHT x WIDTH]
        image_index = 0
        for rgb_imgs, irn_imgs, video_id, frame_id in noisy_img_loader:

            # fpn_imgs: K x [2 x HEIGHT x WIDTH]
            if opt.keep_fpn == True:
                # calculate fpn for the first frame only, use the same fpn along the video
                if frame_id == 1:
                    print("Calculating the FPN using the first frame of %d..." % video_id)
                    fpn_imgs = model.forward(irn_imgs, rgb_imgs)
                    # for K images forward returns K different FPN, we need to use one
                    fpn_imgs = [torch.stack(fpn_imgs, dim=0).mean(dim=0)] * len(fpn_imgs)
            else:
                # calculate fpn for each frame
                fpn_imgs = model.forward(irn_imgs, rgb_imgs)

            # ire_imgs: K x [1 x HEIGHT x WIDTH]
            ire_imgs = [fpn_remove(irn_imgs[idx], fpn_imgs[idx].unsqueeze(0)).squeeze(0) for idx in range(len(fpn_imgs))]

            # evaluate the result of the clean image
            metric_evaluator.evaluate(None, torch.stack(ire_imgs, dim=0))

            # put the fpn buffer result into memory
            if opt.save_fpn:
                fpn_buffer.evaluate(fpn_imgs[0])

            # save the every 20th image
            if image_index % 10 == 0:
                visualize.save_fpn_results(model_name, image_index, fpn_imgs[0], fpn_imgs[0])
                visualize.save_inference_results(model_name, image_index, rgb_imgs[0], irn_imgs[0], irn_imgs[0], ire_imgs[0])
            
            # evaluate the result of the clean image
            image_index += 1

        # save the metric to given file
        metric_evaluator.save_metrics(reduction="mean")
        fpn_buffer.save(model_name)
            
        
    
    