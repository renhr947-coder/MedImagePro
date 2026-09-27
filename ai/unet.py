import torch
import torch.nn as nn



class DoubleConv(nn.Module):

    def __init__(self,in_c,out_c):

        super().__init__()

        self.conv=nn.Sequential(

            nn.Conv2d(
                in_c,
                out_c,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_c),

            nn.ReLU(inplace=True),


            nn.Conv2d(
                out_c,
                out_c,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_c),

            nn.ReLU(inplace=True)

        )


    def forward(self,x):

        return self.conv(x)



class UNet(nn.Module):


    def __init__(self):

        super().__init__()


        self.enc1=DoubleConv(
            1,
            16
        )


        self.pool1=nn.MaxPool2d(2)


        self.enc2=DoubleConv(
            16,
            32
        )


        self.pool2=nn.MaxPool2d(2)


        self.enc3=DoubleConv(
            32,
            64
        )


        self.up1=nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )


        self.dec1=DoubleConv(
            64,
            32
        )


        self.up2=nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )


        self.dec2=DoubleConv(
            32,
            16
        )


        self.out=nn.Conv2d(
            16,
            1,
            1
        )



    def forward(self,x):


        x1=self.enc1(x)


        x2=self.pool1(x1)


        x3=self.enc2(x2)


        x4=self.pool2(x3)


        x5=self.enc3(x4)


        x=self.up1(x5)


        x=self.dec1(x)


        x=self.up2(x)


        x=self.dec2(x)


        x=self.out(x)


        return torch.sigmoid(x)