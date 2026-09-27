import cv2



def gaussian_denoise(image):


    gray=cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    result=cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )


    return result




def median_denoise(image):

    """
    中值滤波去噪
    """

    result = cv2.medianBlur(
        image,
        5
    )


    return result
