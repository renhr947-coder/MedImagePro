import torch
import cv2
import numpy as np

from ai.unet import UNet



def remove_small_regions(mask, min_area=50):
    """
    去除小面积预测区域
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )


    clean_mask = np.zeros_like(mask)


    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]


        if area >= min_area:

            clean_mask[labels == i] = 255


    return clean_mask





def ai_predict(image_path):


    device = "cpu"



    # ==========================
    # 加载模型
    # ==========================

    model = UNet().to(device)


    model.load_state_dict(

        torch.load(

            "weights/unet_best.pth",

            map_location=device

        )

    )


    model.eval()



    # ==========================
    # 读取MRI
    # ==========================

    img = cv2.imread(

        image_path,

        cv2.IMREAD_GRAYSCALE

    )


    if img is None:

        raise FileNotFoundError(
            "无法读取图像: "+image_path
        )


    h,w = img.shape



    # ==========================
    # 预处理
    # ==========================

    img_resize = cv2.resize(

        img,

        (256,256)

    )


    img_resize = img_resize / 255.0



    tensor = torch.tensor(

        img_resize,

        dtype=torch.float32

    )



    tensor = tensor.unsqueeze(0)

    tensor = tensor.unsqueeze(0)



    # ==========================
    # U-Net预测
    # ==========================

    with torch.no_grad():


        pred = model(
            tensor
        )



    probability = pred.squeeze().numpy()



    # ==========================
    # 阈值分割
    # 提高到0.6减少误检
    # ==========================

    mask = (

        probability > 0.6

    ).astype(np.uint8) * 255




    # ==========================
    # 形态学优化
    # ==========================


    kernel = np.ones(

        (3,3),

        np.uint8

    )


    # 去除噪声

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_OPEN,

        kernel

    )


    # 填补空洞

    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel

    )



    # ==========================
    # 连通域过滤
    # 删除小碎片
    # ==========================

    mask = remove_small_regions(

        mask,

        min_area=50

    )



    # ==========================
    # 恢复原始尺寸
    # ==========================

    mask = cv2.resize(

        mask,

        (w,h),

        interpolation=cv2.INTER_NEAREST

    )



    return mask