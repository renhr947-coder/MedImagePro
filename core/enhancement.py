import cv2
import numpy as np


def enhance_image(image):


    # 转灰度

    gray=cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    clahe=cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )


    enhanced=clahe.apply(
        gray
    )


    return enhanced


def sharpen_image(image):

    """
    图像锐化
    """


    kernel=np.array(
        [
            [0,-1,0],
            [-1,5,-1],
            [0,-1,0]
        ]
    )


    result=cv2.filter2D(
        image,
        -1,
        kernel
    )


    return result
