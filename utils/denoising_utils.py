import os
import numpy as np
import cv2 as cv
from scipy.misc import *
import matplotlib.pyplot as plt
import matplotlib
from .common_utils import *


        
def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img = np_to_pil(img_noisy_np)
    img.save('noise.png')  #保存
    # cv2保存图片
    # img_noise = np.transpose(img_noisy_np,(1,2,0))
    # cv.imwrite('noise.png', img_noise*255)
    img_noisy_pil = np_to_pil(img_noisy_np)
    # plot_image_grid([img_noisy_np], 8, 8)
    return img_noisy_pil, img_noisy_np


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)