import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import MRIDataset

from unet import UNet



device="cpu"



dataset=MRIDataset(
    "../dataset/filtered"
)



loader=DataLoader(

    dataset,

    batch_size=4,

    shuffle=True

)



model=UNet().to(device)



bce_loss=nn.BCELoss()



def dice_loss(pred,target):


    smooth=1e-5


    intersection=(pred*target).sum()


    dice=(

        2*intersection+smooth

    )/(

        pred.sum()

        +

        target.sum()

        +

        smooth

    )


    return 1-dice




optimizer=torch.optim.Adam(

    model.parameters(),

    lr=0.001

)



epochs=30



best_loss=999



for epoch in range(epochs):


    total_loss=0


    model.train()


    for image,mask in loader:


        image=image.to(device)

        mask=mask.to(device)



        pred=model(image)



        loss=(

            bce_loss(pred,mask)

            +

            dice_loss(pred,mask)

        )



        optimizer.zero_grad()


        loss.backward()


        optimizer.step()



        total_loss += loss.item()



    avg_loss = total_loss / len(loader)



    print(

        f"Epoch {epoch+1}/{epochs}",

        "Loss:",

        avg_loss

    )



    if avg_loss < best_loss:


        best_loss=avg_loss


        torch.save(

            model.state_dict(),

            "../weights/unet_best.pth"

        )


        print(
            "保存最佳模型"
        )



print(
    "训练完成"
)