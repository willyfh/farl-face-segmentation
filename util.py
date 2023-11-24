import os
import numpy as np
from typing import Dict, List
import torch
import PIL
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import imutils
import cv2


def read_image(path: str, resize=None, ret_numpy_array=False) -> torch.Tensor:
    """Read an image from a given path.
    """
    
    image = Image.open(path)
    if resize and not (resize == [0,0]):
        image = image.resize(resize)
    np_image = np.array(image.convert('RGB'))
    if ret_numpy_array:
        return np_image
    return torch.from_numpy(np_image)

def show_image(image: PIL.Image, title=""):
    plt.figure(title)
    plt.imshow(image)
    plt.show(block=False)

def save_image(image: PIL.Image, input_image_path: str, output_dir: str, target_image_path=None, suffix="mask"):
    os.makedirs(output_dir, exist_ok=True)
    image_name = Path(input_image_path).name
    image_names = image_name.rsplit(".", 1)
    if not target_image_path:
        output_path = Path(output_dir, "{}_{}.{}".format(image_names[0], suffix, image_names[1]))
    else:
        target_name = Path(target_image_path).name
        target_names = target_name.rsplit(".", 1)
        output_path = Path(output_dir, "{}_{}_{}.{}".format(image_names[0], target_names[0], suffix, image_names[1]))
    image = Image.fromarray(image)
    image.save(output_path)
    print("Saved image:", output_path)


