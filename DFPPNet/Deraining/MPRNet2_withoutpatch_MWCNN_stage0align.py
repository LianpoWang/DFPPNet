"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn
from .MWCNN.mwcnn_align import MWCNN_align as MWCNN

#from MWCNN.aligned_model import Aligned



##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 2, kernel_size, bias=bias)
        self.conv3 = conv(2, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img





##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x





##########################################################################
class MPRNetwithoutPatch_withMWCNN_stage0align(nn.Module):
    def __init__(self, in_c=2, out_c=2, n_feat=32, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3,
                 reduction=4, bias=False):
        super(MPRNetwithoutPatch_withMWCNN_stage0align, self).__init__()


        act = nn.PReLU()
        self.MWCNN1 = MWCNN(csff=False,feats = n_feat,stage0_align = True,stage1_align = False)
        self.MWCNN2 = MWCNN(csff=True,feats = n_feat,stage0_align = True,stage1_align = False)
        self.MWCNN3 = MWCNN(csff=True,feats = n_feat,stage0_align = True,stage1_align = False)

        self.sam12 = SAM(n_feat*2, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat*2, kernel_size=1, bias=bias)

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.concat23 = conv(n_feat * 2, n_feat , kernel_size, bias=bias)
        self.tail = conv(n_feat , out_c, kernel_size, bias=bias)

    def forward(self, img_list):
        x3_img_ori = img_list[0]
        x2_img_ori = img_list[1]
        x1_img_ori = img_list[2]
        frame2_ori = img_list[3]
        #ref_frame = frame2_ori[:, 0, :, :].unsqueeze(1)#准备好参考帧
        # x3_img = img_list
        # x2_img = img_list
        # x1_img = img_list
        # frame2 = img_list
        batch_size, num, w, h = frame2_ori.size()
        # center frame interpolation
        center = num // 2
        # extract features
        x3_img = x3_img_ori.unsqueeze(2).view(-1, 1, w, h) # B num H W ->  B num C H W ->B*num C H W  #此处灰度图  通道数为1
        x2_img = x2_img_ori.unsqueeze(2).view(-1, 1, w, h)
        x1_img = x1_img_ori.unsqueeze(2).view(-1, 1, w, h)
        frame2 = frame2_ori.unsqueeze(2).view(-1, 1, w, h)

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        stage1_feature = self.MWCNN1(x1_img)


        # #加入特征对齐模块
        # aligned_feature1 = self.aligned(stage1_feature[5].view(batch_size, num, -1, w, h))
        aligned_feature1 = stage1_feature[5].view(batch_size,-1,w,h)
        ## Apply Supervised Attention Module (SAM)
        x1_samfeats, stage1_img = self.sam12(aligned_feature1, x1_img_ori)

        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        stage2_feature = self.MWCNN2(x2_img,x1_samfeats,stage1_feature)
        #aligned_feature2 = self.aligned(stage2_feature[5].view(batch_size, num, -1, w, h))
        aligned_feature2 = stage2_feature[5].view(batch_size, -1, w, h)
        ## Apply SAM
        x2_samfeats, stage2_img = self.sam23(aligned_feature2, x2_img_ori)

        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        stage3_feature = self.MWCNN3(x3_img,x2_samfeats,stage2_feature)
        #aligned_feature3 = self.aligned(stage3_feature[5].view(batch_size, num, -1, w, h))
        aligned_feature3 = stage3_feature[5].view(batch_size, -1, w, h)
        ## Apply SAM
        x2_samfeats, stage3_img = self.sam23(aligned_feature3, x3_img_ori)
        #stage3_image = self.tail(aligned_feature3)

        # y1 = (stage3_img+x3_img).mean(dim=1, keepdim=True)
        y1 = (stage3_img +  frame2_ori).mean(dim=1, keepdim=True)
        y2 = (stage2_img ).mean(dim=1, keepdim=True)
        y3 = (stage1_img ).mean(dim=1, keepdim=True)

        return [y1, y2, y3]


if __name__ == '__main__':
    import sys
    import os

    sys.path.append(os.path.abspath('.'))

    upscale = 4
    window_size = 16
    # height = (512 // upscale // window_size) * window_size
    # width = (512 // upscale // window_size) * window_size
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = MPRNetwithoutPatch_withMWCNN_stage0align().to('cuda')

    params = sum(map(lambda x: x.numel(), model.parameters()))
    results = dict()
    results[f"runtime"] = []
    model.eval()

    x = torch.randn((2,2, 960, 960)).to('cuda')
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(10):
            x_sr = model([x,x,x,x])
            print("0")