# Face Segmentation of FaRL Model using Facer
A face segmentation implementation of [FarRL](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_General_Facial_Representation_Learning_in_a_Visual-Linguistic_Manner_CVPR_2022_paper.pdf) model (CVPR 2022) using [Facer](https://github.com/FacePerceiver/facer), a face analysis toolkit for modern research.

The author of FaRL performed pre-training on [LAION-FACE](https://github.com/FacePerceiver/LAION-Face), a dataset containing a large amount of face image-text pairs. They then used the pretrained encoder, to extract the image features for training a downstream task, which also includes Face Parsing task. In this work, I utilize the FaRL model which was pretrained on the Face Parsing task, and then map the predicted labels (eyes, nose, mouth, etc.) into two separate labels, i.e., face and background.

## Illustration of the FaRL Pretraining Framework
![FaRL](https://github.com/willyfh/farl-face-segmentation/assets/5786636/b39da57e-ea69-440a-9d17-d9cc6efeab82)

## Installation
```
conda create --name {name_env} python==3.8
conda activate {name_env}
pip install git+https://github.com/FacePerceiver/facer.git@main
pip install timm
pip install imutils
```

## Usage

```
python face_segmentation.py --image_path sample/Shin-Eun-Soo.jpg
```

To display the list of available arguments:
```
python face_segmentation.py --help
```

## Output
For left to right: segmented face image, extracted face mask, cropped face
![Shin-Eun-Soo_stackedsegmentation](https://github.com/willyfh/farl-face-segmentation/assets/5786636/046c3df4-2ec3-4c06-a4c3-cc2e227df93a)
![Shin-Tae-Yong_stackedsegmentation](https://github.com/willyfh/farl-face-segmentation/assets/5786636/d7f9d84a-223c-4c64-bc5b-b07624b1dfac)
![Willy-Fitra-Hendria_stackedsegmentation](https://github.com/willyfh/farl-face-segmentation/assets/5786636/f3146583-8195-45be-96c7-2cbee06551ba)
