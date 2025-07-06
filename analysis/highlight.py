import os
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

# load image, fpn and other parts
def load_image(img_path, model_name, image_idx):
    # load the result image
    imgpx = Image.open(os.path.join(img_path, f"img_{model_name}_{image_idx}.png"))
    imgpx = np.array(imgpx)
    
    # in each image we have 4 images (RGB, IRC, IRN, IRE)
    image_size = imgpx.shape[0]
    rgb = imgpx[:, 0*image_size:1*image_size]
    irc = imgpx[:, 1*image_size:2*image_size]
    irn = imgpx[:, 2*image_size:3*image_size]
    ire = imgpx[:, 3*image_size:4*image_size]
    
    # return all parts
    return rgb, irc, irn, ire

def map_image(input_image):
    darker = (181/255, 215/255, 228/255)
    bright = (197/255, 238/255, 194/255)
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors=[
        (0.0, (0, 0, 1)),
        (0.4, darker),
        (0.5, (0.0, 0.0, 0.0)),
        (0.6, bright),
        (1.0, (0, 1, 0))
    ])

    # get the colormapped imaged
    mapped_image = custom_cmap(np.array(input_image)[:, :, 0] / 255.0)[:, :, :3]
    mapped_image = np.uint8(mapped_image * 255)
    return Image.fromarray(mapped_image)

# crop selected zone and save image
def crop_and_save(img, zoom_img, crop_zone, output_path, model_name, img_id):
    # crop selected region on IRE
    cropped_img = zoom_img[crop_zone[0]:crop_zone[1] + 1, crop_zone[2]:crop_zone[3] + 1]
    zoomed_img = Image.fromarray(cropped_img).resize([ire.shape[1], ire.shape[0]])
    
    # draw horizontal and vertical lines
    img[crop_zone[0]:crop_zone[0] + line_width, crop_zone[2]:crop_zone[3] + 1, :] = color
    img[crop_zone[1]:crop_zone[1] + line_width, crop_zone[2]:crop_zone[3] + 1, :] = color
    img[crop_zone[0]:crop_zone[1] + 1, crop_zone[2]:crop_zone[2] + line_width, :] = color
    img[crop_zone[0]:crop_zone[1] + 1, crop_zone[3]:crop_zone[3] + line_width, :] = color

    # map zoomed image since it is residual
    zoomed_img = map_image(zoomed_img)

    # save image and zoomed image
    bimage = Image.fromarray(img.astype('uint8'))
    bimage.save(os.path.join(output_path, "result_{}_image_{}.png".format(model_name.lower(), img_id)))
    zoomed_img.save(os.path.join(output_path, "zoomed_{}_image_{}_map.png".format(model_name.lower(), img_id)))


# run training code
if __name__ == '__main__':
    # set the output path
    image_path = os.path.join("output", "output_visualizations", "visuals")
    output_path = os.path.join("output", "output_visualizations", "highlights")
    model_names = ["EMPTY", "MULTIVIEW", "DLSNUC", "SAFTA", "SAFTA-RGB", "D1WLS"]
    
    # select highlight images
    highlights = []
    highlights.append({"img_idx": 24, "crop_zone": [25, 125, 118, 228]})
    highlights.append({"img_idx": 45, "crop_zone": [80, 140, 60, 120]})
    

    # highlight color and thickness
    color = [255, 0, 0]
    line_width = 1
    
    # create output path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # all the settings are done, open the image
    for model_id, model_name in enumerate(model_names):
        # get all highlight zones
        for image_index, highlight in enumerate(highlights):
            # get the image
            rgb, irc, irn, ire = load_image(image_path, model_name, highlight["img_idx"])
            
            # crop and save from IRE
            diff = irc.astype(float) - ire.astype(float)
            residual = (255 * (diff + 255) / (2 * 255)) / 0.2
            crop_and_save(ire, residual.astype(np.uint8), highlight["crop_zone"], output_path, model_name, image_index)
        