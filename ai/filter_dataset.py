import os
import cv2
import shutil



source_dir = "../dataset/train_cases"


target_dir = "../dataset/filtered"



os.makedirs(
    target_dir,
    exist_ok=True
)



count = 0



for root, dirs, files in os.walk(source_dir):


    for file in files:


        # 找mask

        if "_mask.tif" in file:


            mask_path = os.path.join(
                root,
                file
            )


            mask = cv2.imread(
                mask_path,
                cv2.IMREAD_GRAYSCALE
            )


            if mask is None:

                print(
                    "读取失败:",
                    mask_path
                )

                continue



            # mask有肿瘤区域

            if mask.sum() > 0:


                image_name = file.replace(
                    "_mask",
                    ""
                )


                image_path = os.path.join(
                    root,
                    image_name
                )


                if os.path.exists(image_path):


                    shutil.copy(

                        image_path,

                        target_dir

                    )


                    shutil.copy(

                        mask_path,

                        target_dir

                    )


                    count += 1


print(
    "筛选完成，共保留:",
    count,
    "张图片"
)
