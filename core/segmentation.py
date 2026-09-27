import cv2
import numpy as np



def threshold_segmentation(image):

    """
    Otsu阈值分割 + 形态学优化
    """


    if len(image.shape)==3:

        gray=cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray=image



    # Otsu

    _,binary=cv2.threshold(

        gray,

        0,

        255,

        cv2.THRESH_BINARY+
        cv2.THRESH_OTSU

    )


    # ---------
    # 形态学优化
    # ---------

    kernel=cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (5,5)

    )


    # 去除小区域

    opening=cv2.morphologyEx(

        binary,

        cv2.MORPH_OPEN,

        kernel

    )


    # 填充区域

    closing=cv2.morphologyEx(

        opening,

        cv2.MORPH_CLOSE,

        kernel

    )


    return closing




def edge_segmentation(image):

    """
    Canny边缘检测
    """


    if len(image.shape)==3:

        gray=cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray=image



    edges=cv2.Canny(

        gray,

        50,

        150

    )


    return edges
