#from .model import common
from Deraining.MWCNN import MWCNN_common as common
import torch
import torch.nn as nn
import scipy.io as sio
from Deraining.MWCNN.aligned_model import Aligned

def make_model(args, parent=False):
    return MWCNN_align(args)

class MWCNN_align(nn.Module):
    def __init__(self, csff = False, feats = 32, conv=common.default_conv,stage0_align = False,stage1_align = False,stage2_align = False):
        super(MWCNN_align, self).__init__()
        self.feats = feats
        n_feats = feats
        kernel_size = 3
        self.scale_idx = 0
        self.stage0_align = stage0_align
        self.stage1_align = stage1_align
        self.stage2_align = stage2_align
        nColor = 1  #图片通道数

        act = nn.ReLU(True)

        self.DWT = common.DWT()
        self.IWT = common.IWT()

        n = 1
        m_head = [common.BBlock(conv, nColor, n_feats, kernel_size, act=act)]
        d_l0 = []
        d_l0.append(common.DBlock_com1(conv, n_feats, n_feats, kernel_size, act=act, bn=False))


        d_l1 = [common.BBlock(conv, n_feats * 4, n_feats * 2, kernel_size, act=act, bn=False)]
        d_l1.append(common.DBlock_com1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False))

        d_l2 = []
        d_l2.append(common.BBlock(conv, n_feats * 8, n_feats * 4, kernel_size, act=act, bn=False))
        d_l2.append(common.DBlock_com1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False))
        pro_l3 = []
        pro_l3.append(common.BBlock(conv, n_feats * 16, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(common.DBlock_com(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(common.DBlock_inv(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(common.BBlock(conv, n_feats * 8, n_feats * 16, kernel_size, act=act, bn=False))

        i_l2 = [common.DBlock_inv1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False)]
        i_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 8, kernel_size, act=act, bn=False))

        i_l1 = [common.DBlock_inv1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False)]
        i_l1.append(common.BBlock(conv, n_feats * 2, n_feats * 4, kernel_size, act=act, bn=False))

        i_l0 = [common.DBlock_inv1(conv, n_feats, n_feats, kernel_size, act=act, bn=False)]

        m_tail = [conv(n_feats, nColor, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)
        self.tail = nn.Sequential(*m_tail)

        self.aligned0 = Aligned(feats=n_feats)
        self.aligned1 = Aligned(feats=n_feats*2)
        self.aligned2 = Aligned(feats=n_feats*4)
        #self.aligned3 = Aligned(feats=n_feats*8)
        self.concat = conv(n_feats *2, n_feats, kernel_size)
        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc0 = nn.Conv2d(n_feats, n_feats, kernel_size=1, bias=False)
            self.csff_enc1 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=1, bias=False)
            self.csff_enc2 = nn.Conv2d(n_feats * 4, n_feats * 4, kernel_size=1, bias=False)

    def forward(self, x,SAM_feature = None,stage_out = None):
        x_shallow = self.head(x)
        BandNum ,_, h, w = x.size()
        # 这里应该拼接 前一阶段送过来的SAM注意力图
        if SAM_feature is not None:
            SAM_feature1 = SAM_feature.view(-1, self.feats, h, w)
            x_shallow = torch.cat([x_shallow, SAM_feature1], 1)
            x_shallow = self.concat(x_shallow)
        x0 = self.d_l0(x_shallow)
        if self.stage0_align:
            x0_aligned = self.aligned0(x0.view(BandNum//2,2,-1,960,960))
            x0 = x0_aligned.view(BandNum,-1,960,960)

        #融合前一阶段送过来的特征
        if (stage_out is not None):
            x0 = x0 + stage_out[0] + stage_out[5]

        x1 = self.d_l1(self.DWT(x0))

        if self.stage1_align:
            x1_aligned = self.aligned1(x1.view(BandNum//2,2,-1,480,480))
            x1 = x1_aligned.view(BandNum,-1,480,480)

        if (stage_out is not None):
            x1 = x1 + stage_out[1] + stage_out[4]

        x2 = self.d_l2(self.DWT(x1))
        if self.stage2_align:
            x2_aligned = self.aligned2(x2.view(BandNum//2,2,-1,240,240))
            x2 = x2_aligned.view(BandNum,-1,240,240)

        if (stage_out is not None):
            x2 = x2 + stage_out[2] + stage_out[3]

        x_2 = self.IWT(self.pro_l3(self.DWT(x2)))+x2

        x_1 = self.IWT(self.i_l2(x_2)) + x1
        x_0 = self.IWT(self.i_l1(x_1)) + x0

        x_img = self.tail(self.i_l0(x_0)) + x

        return [x0,x1,x2,x_2,x_1,x_0,x_img]

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


def main():
    # 使用 option.py 中的 args 初始化模型
    model = MWCNN_align().to('cuda')


    # 打印模型结构（如果需要）
    # if args.print_model:
    #     print(model)

    # 随机生成一个输入张量 (batch_size, n_colors, height, width)
    input_tensor = torch.randn(1, 1, 960, 960).to('cuda')
    SAM_tensor = torch.randn(4,1, 960, 960).to('cuda')

    # 将输入张量传入模型
    output = model(input_tensor)
    # 打印输入和输出的形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == '__main__':
    main()

