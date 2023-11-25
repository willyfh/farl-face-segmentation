import argparse
import facer
import os
import numpy as np
import cv2

import torch
from util import read_image, show_image, save_image
import functools
from facer.transform import (get_face_align_matrix,
                         make_inverted_tanh_warp_grid, make_tanh_warp_grid)

import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List

def get_args():
    parser = argparse.ArgumentParser(description="face segmentation")
    parser.add_argument("--cuda", action='store_true', help="Whether to use cuda or cpu.")
    parser.add_argument("--image_path", type=str, help="path to the input image", required=True)
    parser.add_argument("--output_dir", type=str, help="directory to store the output image", default="output/face_segmentation/")
    parser.add_argument("--face_detector_conf_name", type=str,
        help="configuration name for face detector in facer. Eg. retinaface/resnet50 or retinaface/mobilenet", default="retinaface/resnet50")
    parser.add_argument("--face_parser_conf_name", type=str,
        help="configuration name for face parser in facer. Eg. farl/celebm/448 or farl/lapa/448", default="farl/celebm/448")
    parser.add_argument("--show_image", action='store_true', help="Whether to show image or not.")
    parser.add_argument("--resize_image", nargs=2, type=int, help="resize image to the size of (width, height)", default=[1024,1024])
    parser.add_argument("--face_color", nargs=3, type=int, help="face color segmentation", default=[255, 129, 54])
    args = parser.parse_args()
    return args

# this is to map each of the face parsing label for face segmentation label > 0: background, 1: face
face_label_mapping = {
    'background': 0,
    'neck': 0,
    'face': 1,
    'cloth': 0,
    'rr': 0,
    'lr': 0,
    'rb': 1,
    'lb': 1,
    're': 1,
    'le': 1,
    'nose': 1,
    'imouth': 1,
    'llip': 1,
    'ulip': 1,
    'hair': 0,
    'eyeg': 1,
    'hat': 0,
    'earr': 0,
    'neckl': 0
}

facer.face_parsing.farl.pretrain_settings['celebm/448'] = {
        'url': [
            'https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.celebm.main_ema_181500_jit.pt',
        ],
        'matrix_src_tag': 'points',
        'get_matrix_fn': functools.partial(get_face_align_matrix,
                                           target_shape=(448, 448), target_face_scale=0.8),
        'get_grid_fn': functools.partial(make_tanh_warp_grid,
                                         warp_factor=0.0, warped_shape=(448, 448)),
        'get_inv_grid_fn': functools.partial(make_inverted_tanh_warp_grid,
                                             warp_factor=0.0, warped_shape=(448, 448)),
        'label_names': ['background', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're',
                        'le', 'nose', 'imouth', 'llip', 'ulip', 'hair',
                        'eyeg', 'hat', 'earr', 'neckl']
    }

def crop_face(input_image: np.array, mask_image: np.array) -> np.array:
    cropped_image = cv2.multiply(np.array(input_image), (mask_image/255).astype(np.uint8))
    return cropped_image

def map_face_label(seg_labels: torch.Tensor, data: Dict[str, torch.Tensor]):
    """ Map the face parsing label to face segmentation label (0: background, 1: face)
    """
    for i in range(len(seg_labels)):
        for j in range(len(seg_labels[i])):
            seg_labels[i][j] = face_label_mapping[data['seg']['label_names'][seg_labels[i][j]]]

def extract_face_mask(data: Dict[str, torch.Tensor]) -> np.array:
    seg_logits = data['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w

    predicted_labels = seg_probs.argmax(dim=1).int()

    map_face_label(predicted_labels[0], data)

    mask_image = (predicted_labels*255)
    mask_image = mask_image.permute(1, 2, 0) # c x h x w -> h x w x c
    mask_image = mask_image.repeat(1,1,3)
    mask_image = mask_image.to(torch.uint8).cpu().numpy()
    return mask_image

def segment_input_image(input_image: np.array, mask_image: np.array, face_color: List[int]) -> (np.array,np.array):
    non_face_image = cv2.multiply(np.array(input_image), (1-(mask_image/255)).astype(np.uint8))
    cropped_face_image = crop_face(input_image, mask_image)
    seg_image = cv2.addWeighted(cropped_face_image, 0.5, ((mask_image/255)*face_color).astype(np.uint8), 0.5, 0)
    seg_image = seg_image + non_face_image
    return seg_image, cropped_face_image

if __name__ == "__main__":

    args = get_args()

    device = "cpu"
    if args.cuda:
        device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    # read the image
    input_image = read_image(args.image_path, resize=args.resize_image)
    image = facer.hwc2bchw(input_image).to(device=device)  # image: 1 x 3 x h x w

    # load the pretrained models
    face_detector = facer.face_detector(args.face_detector_conf_name, device=device)
    face_parser = facer.face_parser(args.face_parser_conf_name, device=device)

    # detect the face
    with torch.inference_mode():
        faces = face_detector(image)

    # predict the segmentation
    with torch.inference_mode():
        faces = face_parser(image, faces)

    mask_image = extract_face_mask(faces)
    save_image(mask_image, args.image_path, args.output_dir, suffix="mask")

    seg_image, cropped_face_image = segment_input_image(np.array(input_image), mask_image, args.face_color)
    save_image(seg_image, args.image_path, args.output_dir, suffix="segmented")  
    save_image(cropped_face_image, args.image_path, args.output_dir, suffix="cropped")

    stacked_image = np.hstack([seg_image, mask_image, cropped_face_image])
    save_image(stacked_image, args.image_path, args.output_dir, suffix="stackedsegmentation")

    if args.show_image:
        show_image(stacked_image, "segmented face - face mask - cropped face")

    input("\nExecution is finished. Press enter to quit...")
