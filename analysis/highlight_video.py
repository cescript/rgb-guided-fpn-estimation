import os
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# load image, fpn and other parts
def load_image(dataset, model_name, image_idx):
    image_path = os.path.join("output", dataset, "visuals")
    # load the result image
    imgpx = Image.open(os.path.join(image_path, f"img_{model_name}_{image_idx}.png"))
    imgpx = np.array(imgpx)
    
    # in each image we have 4 images (RGB, IRC, IRN, IRE)
    image_size = imgpx.shape[1] // 4
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

def grayscale_to_thermal(pil_gray, cmap_name="turbo"):
    gray = np.array(pil_gray, dtype=np.uint8)      # H x W x 3
    gray = gray[:, :, 0]
    norm = gray.astype(np.float32) / 255.0          # [0,1]

    cmap = cm.get_cmap(cmap_name)
    colored = cmap(norm)[:, :, :3]                  # drop alpha

    colored = (colored * 255).astype(np.uint8)
    return Image.fromarray(colored, mode="RGB")

# crop selected zone and save image
def crop_and_save(img, zoom_img, crop_zone, output_path, model_name, img_id, save_result):
    # crop selected region on IRE
    cropped_img = zoom_img[crop_zone[0]:crop_zone[1] + 1, crop_zone[2]:crop_zone[3] + 1]
    zoomed_img = Image.fromarray(cropped_img).resize([zoom_img.shape[1], zoom_img.shape[0]])
    
    # draw horizontal and vertical lines
    img[crop_zone[0]:crop_zone[0] + line_width, crop_zone[2]:crop_zone[3] + 1, :] = color
    img[crop_zone[1]:crop_zone[1] + line_width, crop_zone[2]:crop_zone[3] + 1, :] = color
    img[crop_zone[0]:crop_zone[1] + 1, crop_zone[2]:crop_zone[2] + line_width, :] = color
    img[crop_zone[0]:crop_zone[1] + 1, crop_zone[3]:crop_zone[3] + line_width, :] = color

    # map zoomed image since it is residual
    zoomed_img = grayscale_to_thermal(zoomed_img)

    # save image and zoomed image
    bimage = Image.fromarray(img.astype('uint8'))
    if save_result:
        bimage.save(os.path.join(output_path, "result_{}_image_{}.png".format(model_name.lower(), img_id)))
        zoomed_img.save(os.path.join(output_path, "zoomed_{}_image_{}_map.png".format(model_name.lower(), img_id)))

    return bimage, zoomed_img


# save multiple images as grid
def save_multi_grid_(image_sets, out_path):
    n = len(image_sets)
    w, h = image_sets[0]["irn"].size
    grid = Image.new("RGB", (2 * n * w, 3 * h))
    for i, imgs in enumerate(image_sets):
        x0 = 2 * i * w
        grid.paste(imgs["irn"],   (x0,     0))
        grid.paste(imgs["irn_z"], (x0 + w, 0))
        grid.paste(imgs["ire"],   (x0,     h))
        grid.paste(imgs["ire_z"], (x0 + w, h))
        grid.paste(imgs["rgb"],   (x0,   2*h))
        grid.paste(imgs["rgb_z"], (x0 + w, 2*h))
    grid.save(out_path)

def save_multi_grid(image_sets, out_path, padding=5):
    n = len(image_sets)
    w, h = image_sets[0]["irn"].size

    rows = 3
    cols = 2 * n
    grid_w = cols * w + (cols - 1) * padding
    grid_h = rows * h + (rows - 1) * padding

    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    for i, imgs in enumerate(image_sets):
        col0 = 2 * i

        x_orig = col0 * (w + padding)
        x_zoom = (col0 + 1) * (w + padding)

        grid.paste(imgs["irn"],   (x_orig, 0))
        grid.paste(imgs["irn_z"], (x_zoom, 0))

        y = h + padding
        grid.paste(imgs["ire"],   (x_orig, y))
        grid.paste(imgs["ire_z"], (x_zoom, y))

        y = 2 * (h + padding)
        grid.paste(imgs["rgb"],   (x_orig, y))
        grid.paste(imgs["rgb_z"], (x_zoom, y))

    grid.save(out_path)

# run training code
if __name__ == '__main__':
    # set the output path
    output_path = os.path.join("output", "highlights_video")

    # select highlight images
    highlights = []
    video_name = "davis_cas_2"
    highlights.append({"dataset": "butiv", "img_idx": 0, "crop_zone": [216, 296, 270, 370]})
    highlights.append({"dataset": "butiv", "img_idx": 50, "crop_zone": [216, 296, 270, 370]})
    highlights.append({"dataset": "butiv", "img_idx": 100, "crop_zone": [216, 296, 270, 370]})

    # highlight color and thickness
    color = [255, 0, 0]
    line_width = 1
    
    # create output path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # all the settings are done, open the image
    # get all highlight zones
    image_sets = []
    for image_index, highlight in enumerate(highlights):
        # get the image
        rgb, irc, irn, ire = load_image(highlight["dataset"], video_name, highlight["img_idx"])

        # crop and save from IRE
        [ire, ire_z] = crop_and_save(ire, ire, highlight["crop_zone"], output_path, f"ire_{video_name}", image_index, False)
        [irn, irn_z] = crop_and_save(irn, irn, highlight["crop_zone"], output_path, f"irn_{video_name}", image_index, False)
        [rgb, rgb_z] = crop_and_save(rgb, rgb, highlight["crop_zone"], output_path, f"med_{video_name}", image_index, False)

        image_sets.append({
            "irn": irn, "irn_z": irn_z,
            "ire": ire, "ire_z": ire_z,
            "rgb": rgb, "rgb_z": rgb_z
        })

    # save multiple images at once
    save_multi_grid(image_sets, f"{output_path}/grid_{video_name}.png")
