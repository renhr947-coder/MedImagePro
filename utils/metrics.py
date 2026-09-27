import numpy as np



def calculate_dice(pred, true):


    pred = pred > 0

    true = true > 0



    intersection = np.logical_and(
        pred,
        true
    ).sum()



    dice = (

        2.0 * intersection

        /

        (
            pred.sum()
            +
            true.sum()
            +
            1e-8
        )

    )


    return dice





def calculate_iou(pred,true):


    pred=pred>0

    true=true>0



    intersection=np.logical_and(
        pred,
        true
    ).sum()



    union=np.logical_or(
        pred,
        true
    ).sum()



    return intersection/(union+1e-8)