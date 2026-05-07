import os
import torch
import models
import argparse
from torchvision.utils import save_image
from utility.FPNUtility import fpn_insert, fpn_remove

# import required classes
from dataloader.DetectionDataLoader import DetectionDataLoader

# get the test configuration
def get_test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="m3fd_config")
    parser.add_argument('--fpn_type', type=str, default="hfn")
    parser.add_argument('--aggregation_size', type=int, default=12)
    return parser.parse_args()

# run test code
if __name__ == '__main__':

    # get the test configuration
    opt = get_test_config()

    # models to evaluate
    model_names = ["SAFTA-RGB", "DCGAN", "MULTIVIEW", "D1WLS", "DLSNUC"]

    # create output directories
    top_folder = os.path.join("output", "detection", f"{opt.dataset}_{opt.fpn_type}")
    clean_dir = os.path.join(top_folder, "CLEAN")
    noisy_dir = os.path.join(top_folder, "NOISY")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    model_dirs = {}
    for model_name in model_names:
        model_dirs[model_name] = os.path.join(top_folder, model_name)
        os.makedirs(model_dirs[model_name], exist_ok=True)

    # loop over all models and test the results
    for mid, model_name in enumerate(model_names):
        print("Evaluating %s model..." % model_name)

        # create a model given opt.model and other options
        model = models.GenerateModel(model_name)

        # recreate the loader for each model (same random seed ensures same data)
        det_loader = DetectionDataLoader(opt.dataset, opt.aggregation_size, opt.fpn_type)

        # start the image loader
        # fpn_img: 1 x 2 x H x W, rgb_img: K x 3 x H x W, irc_img: K x 1 x H x W
        image_index = 0
        for fpn_img, rgb_img, irc_img in det_loader:
            # get the filename stem for this test image
            stem = det_loader.get_image_stem(image_index)

            # irn_imgs: K x [1 x H x W]
            # rgb_imgs: K x [3 x H x W]
            irn_imgs = [fpn_insert(irc_img[idx], fpn_img).squeeze(0) for idx in range(irc_img.shape[0])]
            rgb_imgs = [rgb_img[idx] for idx in range(rgb_img.shape[0])]

            # fpn_imgs: K x [2 x H x W]
            fpn_imgs = model.forward(irn_imgs, rgb_imgs)

            # ire_imgs: K x [1 x H x W]
            ire_imgs = [fpn_remove(irn_imgs[idx], fpn_imgs[idx].unsqueeze(0)).squeeze(0) for idx in range(len(fpn_imgs))]

            # save clean and noisy only on the first model run
            if mid == 0:
                save_image(irc_img[0], os.path.join(clean_dir, f"{stem}.png"))
                save_image(irn_imgs[0], os.path.join(noisy_dir, f"{stem}.png"))

            # save denoised target image (index 0)
            save_image(ire_imgs[0], os.path.join(model_dirs[model_name], f"{stem}.png"))

            image_index += 1

        print("  saved %d images to %s" % (image_index, model_dirs[model_name]))

    print("detection dataset generation complete: %s" % top_folder)