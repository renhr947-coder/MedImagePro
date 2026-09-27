import cv2
import numpy as np



def calculate_features(image, mask):

    """
    医学图像特征分析

    image:
        原始医学图像

    mask:
        分割结果
    """


    # 图像尺寸

    height, width = image.shape[:2]


    total_pixels = height * width



    # 分割区域面积

    area = np.sum(
        mask > 0
    )



    # 区域比例

    ratio = area / total_pixels * 100



    # 如果是RGB

    if len(image.shape)==3:

        gray=cv2.cvtColor(

            image,

            cv2.COLOR_BGR2GRAY

        )

    else:

        gray=image



    # 分割区域灰度

    region_pixels = gray[
        mask > 0
    ]



    if len(region_pixels)>0:


        mean_intensity = np.mean(
            region_pixels
        )


        max_intensity = np.max(
            region_pixels
        )


        min_intensity = np.min(
            region_pixels
        )


    else:


        mean_intensity=0

        max_intensity=0

        min_intensity=0



    result={


        "Image Size":

        f"{width} × {height}",



        "Segmentation Area":

        f"{area} pixels",



        "Region Ratio":

        f"{ratio:.2f} %",



        "Mean Intensity":

        f"{mean_intensity:.2f}",



        "Maximum Intensity":

        str(max_intensity),



        "Minimum Intensity":

        str(min_intensity)

    }



    return result
def save_report(features,
                filename="results/Medical_Report.txt"):


    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:


        f.write(
            "================================\n"
        )


        f.write(
            "MedImagePro Analysis Report\n"
        )


        f.write(
            "================================\n\n"
        )


        for key,value in features.items():


            f.write(

                key
                +
                ": "
                +
                value
                +
                "\n"

            )
