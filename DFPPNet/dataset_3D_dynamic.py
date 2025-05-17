import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import torchvision.transforms as transforms
import random

from PIL import Image, ImageEnhance

def adjust_brightness_contrast(image, brightness_factor, contrast_factor):
    # 调整亮度
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    # 调整对比度
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    return image

class MyDataset(Dataset):
    def __init__(self, root_dir,gt_truth,data_factor, transform=None):
        self.root_dir = root_dir
        self.gt_truth = gt_truth
        self.data_factor = data_factor
        self.all_folders = [os.path.join(root_dir, folder) for folder in os.listdir(root_dir)]
        self.all_subfolders = []
        for folder in self.all_folders:
            self.all_subfolders.extend([os.path.join(folder, subfolder) for subfolder in os.listdir(folder)])
        self.transform = transform  # 传入的transform应该只包含ToTensor等不涉及随机性的转换
        self.is_cropped = False

    def __len__(self):
        return len(self.all_subfolders)

    def __getitem__(self, idx):
        subfolder_path = self.all_subfolders[idx]
        real_phase_path = os.path.join(subfolder_path, 'real_phase')
        images = []
        tran_random = random.random()
        # 定义转换
        transform1 = transforms.ToTensor()
        for file in sorted(os.listdir(real_phase_path)):
            if file.endswith('.bmp'):
                image_path = os.path.join(real_phase_path, file)
                image = Image.open(image_path)
                image = image.crop((0, 0, 960, 960))  # 只裁剪一次

                # 应用随机翻转
                if 0 <tran_random < 0.33:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                elif 0.33<=tran_random<0.66:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)

                # 先转换为RGB图
                #image = image.convert('RGB')
                if self.transform:
                    image = self.transform(image)  # 应用其他转换

                # 然后转换为灰度图
                image = image.convert('L')

                image = transform1(image)
                images.append(image)

        images = torch.stack(images)  # 使用 torch.stack 以保持维度一致

        csv_path = os.path.join(real_phase_path,self.gt_truth)
        label = pd.read_csv(csv_path, header=None, index_col=False).values
        label = label / self.data_factor
        label = Image.fromarray(label)
        label = label.crop((0, 0, 960, 960))  # 中心裁剪

        # 如果图像被翻转，标签也翻转
        if 0 < tran_random < 0.33:
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        elif 0.33 <= tran_random < 0.66:
            label = label.transpose(Image.FLIP_TOP_BOTTOM)


        label = transform1(label)
        self.is_cropped = True  # 只裁剪一次
        #label = label / 150.0
        return images, label
