import cv2
import numpy as np



def create_overlay(image, mask):


    # 灰度转RGB

    if len(image.shape)==2:

        image_color=cv2.cvtColor(

            image,

            cv2.COLOR_GRAY2BGR

        )

    else:

        image_color=image.copy()



    # mask膨胀，让区域更加明显

    kernel=cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (3,3)

    )


    mask=cv2.dilate(

        mask,

        kernel,

        iterations=1

    )



    # 创建红色区域

    color_mask=np.zeros_like(

        image_color

    )


    color_mask[:,:,2]=255



    # 只保留mask区域

    color_mask = color_mask * (

        mask[:,:,None] > 0

    )



    # 半透明融合

    overlay=cv2.addWeighted(

        image_color,

        0.7,

        color_mask,

        0.3,

        0

    )


    return overlay
