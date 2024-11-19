from spectral import *
import numpy as np
from sklearn.decomposition import PCA
from patchify import patchify
import torch
from torch import nn
import cv2


class Generator(nn.Module):
    def __init__(self,in_channels,features):
        super(Generator,self).__init__()

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.down1 = nn.Sequential(
            # C-128
            nn.Conv2d(features,features*2,(4,4),(2,2),padding=(1,1)),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down2 = nn.Sequential(
            # C-256
            nn.Conv2d(features*2,features*4,(4,4),(2,2),padding=(1,1)),
            nn.InstanceNorm2d(num_features=features*4),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down3 = nn.Sequential(
            # C-512
            nn.Conv2d(features*4,features*8,(4,4),(2,2),padding=(1,1)),
            nn.InstanceNorm2d(num_features=features*8),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down4 = nn.Sequential(
            # C-512
            nn.Conv2d(features*8,features*8,(4,4),(2,2),padding=(1,1)),
            nn.InstanceNorm2d(num_features=features*8),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down5 = nn.Sequential(
            # C-512
            nn.Conv2d(features*8,features*8,(4,4),(2,2),padding=(1,1)),
            nn.InstanceNorm2d(num_features=features*8),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down6 = nn.Sequential(
            # C-512
            nn.Conv2d(features*8,features*8,(4,4),(2,2),padding=(1,1)),
            nn.InstanceNorm2d(num_features=features*8),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down7 = nn.Sequential(
            # C-512
            nn.Conv2d(features*8,features*8,(4,4),(2,2),padding=(1,1)),
            nn.InstanceNorm2d(num_features=features*8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8,features*8,(4,4),(2,2),padding=(1,1)),
            nn.ReLU()
        )

        self.up1 = nn.Sequential(
            # C-512
            nn.ConvTranspose2d(features*8,features*8,(4,4),(2,2),padding=(1,1)),
            nn.InstanceNorm2d(num_features=features*8),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            # C-512
            nn.ConvTranspose2d(features*8*2,features*8,(4,4),(2,2),padding=(1,1)),
            nn.InstanceNorm2d(num_features=features*8),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            # C-512
            nn.ConvTranspose2d(features*8*2,features*8,(4,4),(2,2),padding=(1,1)),
            nn.InstanceNorm2d(num_features=features*8),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            # C-512
            nn.ConvTranspose2d(features*8*2,features*8,(4,4),(2,2),padding=(1,1)),
            nn.InstanceNorm2d(num_features=features*8),
            nn.ReLU()
        )
        self.up5 = nn.Sequential(
            # C-256
            nn.ConvTranspose2d(features*8*2,features*8,(4,4),(2,2),padding=(1,1)),
            nn.InstanceNorm2d(num_features=features*8),
            nn.ReLU()
        )
        self.up6 = nn.Sequential(
            # C-256
            nn.ConvTranspose2d(features*8*2,features*4,(4,4),(2,2),padding=(1,1)),
            nn.InstanceNorm2d(num_features=features*4),
            nn.ReLU()
        )
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(features*4*2,features*2,(4,4),(2,2),padding=(1,1)),
            nn.InstanceNorm2d(num_features=features*2),
            nn.ReLU()
        )
        self.up8 = nn.Sequential(
            # C-256
            nn.ConvTranspose2d(features*2*2,features,(4,4),(2,2),padding=(1,1)),
            nn.InstanceNorm2d(num_features=features),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.ConvTranspose2d(features*2,in_channels,(4,4),(2,2),padding=(1,1)),
            nn.Tanh()
        )

    def forward(self,x):
        ini = self.initial_down(x)
        d1 = self.down1(ini)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        # apply bottleneck
        b = self.bottleneck(d7)
        # decoder part
        u1 = self.up1(b)
        u2 = self.up2(torch.cat([u1,d7],1))
        u3 = self.up3(torch.cat([u2,d6],1))
        u4 = self.up4(torch.cat([u3,d5],1))
        u5 = self.up5(torch.cat([u4,d4],1))
        u6 = self.up6(torch.cat([u5,d3], 1))
        u7 = self.up7(torch.cat([u6,d2], 1))
        u8 = self.up8(torch.cat([u7,d1],1))
        out = self.output(torch.cat([u8,ini],1))
        return out


# img = torch.randn((1,3,512,512))
# model = Generator(in_channels=3,features=64)
# y_pred = model(img)
# print(y_pred.shape)



#
# import torch
# import torch.nn as nn
#
#
# class Block(nn.Module):
#     def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
#         super(Block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
#             if down
#             else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
#         )
#
#         self.use_dropout = use_dropout
#         self.dropout = nn.Dropout(0.5)
#         self.down = down
#
#     def forward(self, x):
#         x = self.conv(x)
#         return self.dropout(x) if self.use_dropout else x
#
#
# class Generator(nn.Module):
#     def __init__(self, in_channels=3, features=64):
#         super().__init__()
#         self.initial_down = nn.Sequential(
#             nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
#             nn.LeakyReLU(0.2),
#         )
#         self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
#         self.down2 = Block(
#             features * 2, features * 4, down=True, act="leaky", use_dropout=False
#         )
#         self.down3 = Block(
#             features * 4, features * 8, down=True, act="leaky", use_dropout=False
#         )
#         self.down4 = Block(
#             features * 8, features * 8, down=True, act="leaky", use_dropout=False
#         )
#         self.down5 = Block(
#             features * 8, features * 8, down=True, act="leaky", use_dropout=False
#         )
#         self.down6 = Block(
#             features * 8, features * 8, down=True, act="leaky", use_dropout=False
#         )
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
#         )
#
#         self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
#         self.up2 = Block(
#             features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
#         )
#         self.up3 = Block(
#             features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
#         )
#         self.up4 = Block(
#             features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
#         )
#         self.up5 = Block(
#             features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
#         )
#         self.up6 = Block(
#             features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
#         )
#         self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
#         self.final_up = nn.Sequential(
#             nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
#             nn.Tanh(),
#         )
#
#     def forward(self, x):
#         d1 = self.initial_down(x)
#         d2 = self.down1(d1)
#         d3 = self.down2(d2)
#         d4 = self.down3(d3)
#         d5 = self.down4(d4)
#         d6 = self.down5(d5)
#         d7 = self.down6(d6)
#         bottleneck = self.bottleneck(d7)
#         up1 = self.up1(bottleneck)
#         up2 = self.up2(torch.cat([up1, d7], 1))
#         up3 = self.up3(torch.cat([up2, d6], 1))
#         up4 = self.up4(torch.cat([up3, d5], 1))
#         up5 = self.up5(torch.cat([up4, d4], 1))
#         up6 = self.up6(torch.cat([up5, d3], 1))
#         up7 = self.up7(torch.cat([up6, d2], 1))
#         return self.final_up(torch.cat([up7, d1], 1))
#
#
# def test():
#     x = torch.randn((1, 3, 256, 256))
#     model = Generator(in_channels=3, features=64)
#     preds = model(x)
#     print(preds.shape)
#
#
# if __name__ == "__main__":
#     test()
