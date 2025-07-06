import torch

# given an img: 1xHxW and multiple fpn nx2xHxW
# returns noisy imgs [n, 1, H, W]
def fpn_insert(irc_img, fpn_img):
    # get scale and bias terms from fpn
    scale, bias = fpn_img[:, 0], fpn_img[:, 1]
    # generate noisy image using broadcasting
    noisy = (irc_img - bias) / scale
    return torch.clamp(noisy, 0.0, 1.0).unsqueeze(1)

# given a set of noisy images [n, 1, H, W] and an fpn: [1, 2, H, W]
# returns clean images [n, 1, H, W]
def fpn_remove(irn_img, fpn_img):
    scale, bias = fpn_img[:, 0], fpn_img[:, 1]
    irc_img = irn_img.squeeze(1) * scale + bias
    return torch.clamp(irc_img, 0.0, 1.0).unsqueeze(1)
