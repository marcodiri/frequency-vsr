import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy import signal


def normalize_img(x):
    return (x - 0.5) * 2.0


def denormalize_img(x):
    return x * 0.5 + 0.5


def downsample(img):
    w, h = img.size
    img = img.resize((w // 2, h // 2), Image.ANTIALIAS)
    return img


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        normalize_img,
    ]
)

transform_from_np = torchvision.transforms.Compose(
    [
        lambda x: x.permute(2, 1, 0),
        normalize_img,
    ]
)

de_normalize = denormalize_img
de_transform = torchvision.transforms.Compose(
    [de_normalize, torchvision.transforms.ToPILImage()]
)


def get_pics_in_subfolder(path, ext="jpg"):
    return list(Path(path).rglob(f"*.{ext}"))


def parse_frame_title(filename: str):
    file_parts = filename.split("_")
    seq = "_".join(file_parts[:-1])
    _, dim, _, _, _, frm_ext = file_parts
    (hr_w, hr_h) = dim.split("x")
    frm, _ = frm_ext.split(".")
    return seq, (int(hr_w), int(hr_h)), int(frm)


def load_img(path):
    img = Image.open(path)
    return img


def create_kernel(opt):
    sigma = opt["dataset"]["degradation"].get("sigma", 1.5)
    ksize = 1 + 2 * int(sigma * 3.0)

    gkern1d = signal.gaussian(ksize, std=sigma).reshape(ksize, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gaussian_kernel = gkern2d / gkern2d.sum()
    zero_kernel = np.zeros_like(gaussian_kernel)

    kernel = np.float32(
        [
            [gaussian_kernel, zero_kernel, zero_kernel],
            [zero_kernel, gaussian_kernel, zero_kernel],
            [zero_kernel, zero_kernel, gaussian_kernel],
        ]
    )

    device = torch.device(opt["device"])
    kernel = torch.from_numpy(kernel).to(device)

    return kernel


def float32_to_uint8(inputs):
    """Convert np.float32 array to np.uint8

    Parameters:
        :param input: np.float32, (NT)CHW, [0, 1]
        :return: np.uint8, (NT)CHW, [0, 255]
    """
    return np.uint8(np.clip(np.round(inputs * 255), 0, 255))


def canonicalize(data):
    """Convert data to torch tensor with type float32

    Assume data has type np.uint8/np.float32 or torch.uint8/torch.float32,
    and uint8 data ranges in [0, 255] and float32 data ranges in [0, 1]
    """

    if isinstance(data, np.ndarray):
        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0
        data = torch.from_numpy(np.ascontiguousarray(data))

    elif isinstance(data, torch.Tensor):
        if data.dtype == torch.uint8:
            data = data.float() / 255.0

    else:
        raise NotImplementedError()

    return data


def save_sequence(seq_dir, seq_data, frm_idx_lst=None, to_bgr=False):
    """Save each frame of a sequence to .png image in seq_dir

    Parameters:
        :param seq_dir: dir to save results
        :param seq_data: sequence with shape thwc|uint8
        :param frm_idx_lst: specify filename for each frame to be saved
        :param to_bgr: whether to flip color channels
    """

    if to_bgr:
        seq_data = seq_data[..., ::-1]  # rgb2bgr

    # use default frm_idx_lst is not specified
    tot_frm = len(seq_data)
    if frm_idx_lst is None:
        frm_idx_lst = ["{:04d}.png".format(i) for i in range(tot_frm)]

    # save for each frame
    os.makedirs(seq_dir, exist_ok=True)

    import cv2

    for i in range(tot_frm):
        cv2.imwrite(osp.join(seq_dir, frm_idx_lst[i]), seq_data[i])
