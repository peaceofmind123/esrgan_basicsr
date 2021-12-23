import torch
import numpy as np
import cv2

def generateFeatureFromImg(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    img = img * 1.0 / 255
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    # matlab imresize
    img_LR = img.unsqueeze(0)

    return img_LR