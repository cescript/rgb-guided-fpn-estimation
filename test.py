import torch
import models
from metrics import MetricEvaluator
from utility.OutputVisualizer import OutputVisualizer
from utility.FPNUtility import fpn_insert, fpn_remove

# import required classes
from dataloader.NoiseImageDataLoader import NoiseImageDataLoader

# run test code
if __name__ == '__main__':
    # test cases
    dataset_names = ["m3fd_config", "msrs_config", "llvip_config"]
    model_names = ["SAFTA-RGB", "DCGAN", "MULTIVIEW", "D1WLS", "DLSNUC", "EMPTY"]

    # set test options
    aggregation_size = 12
    save_results = 50

    # loop over all cases
    for dataset in dataset_names:
        # create test output name
        output_folder = dataset + "_" + "result_fpn"

        # create a visualizer and metric evaluator
        visualize = OutputVisualizer("output", output_folder, save_logs=True)
        metric_evaluator = MetricEvaluator("output", output_folder)

        # loop over all models and test the results
        for mid, model_name in enumerate(model_names):
            print("Evaluating %s model..." % model_name)

            # create dataset loader and fpn generator for training
            fpn_img_loader = NoiseImageDataLoader(dataset, 1, aggregation_size, is_train_mode=False)

            # start the evaluation of the model
            metric_evaluator.start(model_name)

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
                if image_index <= save_results:
                    visualize.save_fpn_results(model_name, image_index, fpn_img[0], fpn_imgs[0])
                    visualize.save_inference_results(model_name, image_index, rgb_imgs[0], irc_img[0], irn_imgs[0], ire_imgs[0])

                # evaluate the result of the clean image
                metric_evaluator.evaluate(irc_img, torch.stack(ire_imgs, dim=0))
                image_index += 1

            # save the metric to given file
            metric_evaluator.save_metrics(reduction="mean")
        
    
    