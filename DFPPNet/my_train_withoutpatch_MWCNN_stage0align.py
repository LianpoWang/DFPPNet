import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '../dataset/'))
sys.path.append(os.path.join(dir_name, '..'))
print(sys.path)
print(dir_name)

import argparse
import my_options

######### parser ###########
opt = my_options.Options().init(argparse.ArgumentParser(description='Image denoising')).parse_args()
print(opt)

import utils
from new_dataset2 import *

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import random
import time
import numpy as np
import datetime
import visdom
# 导入loss函数
from my_loss import MyL1, MyL2, CustomLoss
# from LAE_GSSE_loss import CombinedLoss

from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.cuda.amp import autocast, GradScaler

from Deraining.MPRNet2_withoutpatch_MWCNN_stage0align import MPRNetwithoutPatch_withMWCNN_stage0align as MPRNet

# from basicsr.archs.craft_arch import CRAFT
# from utils.loader import  get_training_data,get_validation_data
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:21'

print(torch.cuda.is_available())


def tensor_to_heatmap(tensor, dir):
    # 确保输入是 3 维的，即 [B, H, W]
    tensor = torch.squeeze(tensor)

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    # 添加一个通道维度，变为 [B, C, H, W]
    tensor = tensor.unsqueeze(1)

    # 获取批次大小
    batch_size = tensor.size(0)

    # 检查指定目录是否存在，若不存在则创建
    save_dir = os.path.join('./heatmap', dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    heatmaps = []
    for i in range(batch_size):
        # 选取单张图像，并去除批次维度
        single_image_tensor = tensor[i]

        # 将 PyTorch 张量转换为 NumPy 数组
        np_image = single_image_tensor.cpu().numpy()
        np_image = np.squeeze(np_image)

        # 确保数据类型是 float32 或 uint8
        if np_image.dtype != np.float32 and np_image.dtype != np.uint8:
            np_image = np_image.astype(np.float32)

        # 创建热力图
        plt.imshow(np_image, cmap='hot', interpolation='nearest')
        plt.axis('off')

        # 构建保存路径
        heatmap_path = os.path.join(save_dir, f'heatmap_{i}.png')

        # 将 matplotlib 图像保存到指定路径
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # 从文件读取图像并转换为张量
        heatmap_tensor = ToTensor()(Image.open(heatmap_path))
        heatmaps.append(heatmap_tensor)

    # 返回一个包含所有热力图张量的堆叠张量
    return torch.stack(heatmaps)


if __name__ == '__main__':
    ######### Logs dir ###########
    log_dir = os.path.join(opt.save_dir, opt.dataset, opt.Ex_num)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt')
    logname = os.path.join(log_dir, 'log.txt')
    print("Now time is : ", datetime.datetime.now().isoformat())
    result_dir = os.path.join(log_dir, 'results')
    model_dir = os.path.join(log_dir, 'models')
    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    # visdom可视化
    vizs = visdom.Visdom(env=opt.Ex_num, port=8887)
    # vizs.line([[0]], [0], win='train', opts=dict(title='loss', legend=['loss']))

    # ######### Set Seeds ###########
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    ######### Model ###########
    # model_restoration = utils.get_arch(opt)
    model_restoration = MPRNet()
    with open(logname, 'a') as f:
        f.write(str(opt) + '\n')
        f.write(str(model_restoration) + '\n')

    ######### Optimizer ###########
    start_epoch = 1
    if opt.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=opt.weight_decay)
    else:
        raise Exception("Error optimizer...")

    print(torch.cuda.is_available())

    ######### DataParallel ###########

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), "GPUs.")
        model_restoration = torch.nn.DataParallel(model_restoration)
    model_restoration.to(device)

    ######### Scheduler ###########
    if opt.warmup:
        print("Using warmup and cosine strategy!")
        warmup_epochs = opt.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - warmup_epochs, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)
        scheduler.step()
    else:
        step = 50
        print("Using StepLR,step={}!".format(step))
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step()

    ######### Resume ###########
    if opt.resume:
        path_chk_rest = opt.pretrain_weights
        print("Resume from " + path_chk_rest)
        utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        lr = utils.load_optim(optimizer, path_chk_rest)
        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_last_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)

    ######### Loss ###########
    # 定义损失函数
    # if opt.loss == "LAE_GSSE_loss":
    #
    #     # criterion = CombinedLoss(alpha=5.0, beta=1.0, gamma=5.0 / 3.0, delta=10.0 / 3.0)
    #     # print("进入")
    # else:
    criterion = CustomLoss()
    # criterion = CharbonnierLoss().cuda()

    ######### DataLoader ###########
    print('===> Loading datasets')
    img_options_train = {'patch_size': opt.train_ps}
    # 在实例化MyDataset时，传入必要的转换
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
    ])

    # 创建数据集和数据加载器
    dataset = MyDataset(opt.data_dir, opt.gt_FrameNum, opt.data_factor, transform=transform)
    # 计算训练集和测试集的大小
    total_size = len(dataset)
    train_size = int(0.8 * total_size)  # 90%的数据用于训练
    val_size = int(0.1 * total_size)  # 90%的数据用于训练
    test_size = total_size - train_size - val_size
    # 随机分割数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # train_dataset = get_training_data(opt.train_dir, img_options_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.train_workers, pin_memory=False, drop_last=False)

    # val_dataset = get_validation_data(opt.val_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

    test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False,
                             num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

    len_trainset = train_dataset.__len__()
    len_valset = val_dataset.__len__()
    len_testset = test_dataset.__len__()
    print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset, ", sizeof test set: ",
          len_testset)

    ######### train ###########
    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.nepoch))
    best_mae = 100
    best_test_mae = 100
    best_epoch = 0
    best_iter = 0
    eval_now = len(train_loader) // 1
    print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

    loss_scaler = NativeScaler()
    # torch.cuda.empty_cache()
    for epoch in range(start_epoch, opt.nepoch + 1):

        # model_restoration.phaseExtract.phase_extraction.update_real_weights()#在每个epoch开始时更新实部权重

        epoch_start_time = time.time()
        mae_train = []
        rmse_train = []
        epoch_loss = 0
        epoch_loss_stage0 = 0
        epoch_loss_stage1 = 0
        epoch_loss_stage2 = 0
        epoch_loss_val = 0
        epoch_loss_test = 0
        train_id = 1
        visdom_restored_list = []
        visdom_target_list = []
        for i, data in enumerate(tqdm(train_loader), 0):

            # zero_grad
            optimizer.zero_grad()
            input_ = [img.to(device) for img in data[0]]
            input_ = [img.squeeze(1) for img in input_]
            #for i, img in enumerate(input_):
                #print(f"Shape of tensor {i+1}: {img.shape}")

            target = data[1].to(device)  # (1,1,720,720)
            #input_ = data[0].to(device)  # (1,4,1,720,720)
            #input_ = input_.squeeze(2)


            if epoch > 5:
                target, input_ = utils.list_MixUp_AUG().aug(target, input_)
            with torch.cuda.amp.autocast():

                restored = model_restoration(input_)

                target_cpu = target.cpu().float()
                # restored_cpu = np.array(restored)

                losses = torch.stack([criterion(restored[j].cpu(), target_cpu) for j in range(len(restored))])

                loss = torch.sum(losses).cuda()
                # loss = criterion(target_cpu, restored_cpu).cuda()
                mae_train.append(utils.batch_MAE(restored[0], target, False).item())
                rmse_train.append(utils.batch_RMSE(restored[0], target, False).item())
            loss_scaler(
                loss, optimizer, parameters=model_restoration.parameters())
            epoch_loss += loss.item()
            epoch_loss_stage0 += losses[0].item()
            epoch_loss_stage1 += losses[1].item()
            epoch_loss_stage2 += losses[2].item()

            # visdom
            if i <= 3:
                # visdom_restored = restored.cpu().numpy()
                # visdom_target = target.cpu().numpy()
                visdom_restored_list.append(tensor_to_heatmap(restored[0].detach(), opt.Ex_num))
                visdom_target_list.append(tensor_to_heatmap(target.detach(), opt.Ex_num))
            if i == 3:
                concatenated_visdom_restored = torch.cat(visdom_restored_list, dim=0)
                concatenated_visdom_target = torch.cat(visdom_target_list, dim=0)
                vizs.images(concatenated_visdom_restored,
                            opts=dict(title='train_restored images', caption='How random.', nrow=4),
                            win="train_restored")
                vizs.images(concatenated_visdom_target,
                            opts=dict(title='train_target images', caption='How random.', nrow=4), win="train_target")

            #### Evaluation ####
            if (i + 1) % eval_now == 0 and i > 0:
                with torch.no_grad():
                    model_restoration.eval()
                    mae_val = []
                    rmse_val = []
                    visdom_restored_list = []
                    visdom_target_list = []
                    for ii, data_val in enumerate((val_loader), 0):
                        target = data_val[1].to(device)
                        # input_ = data_val[0].to(device)
                        # input_ = input_.squeeze(2)

                        input_ = [img.to(device) for img in data_val[0]]
                        input_ = [img.squeeze(1) for img in input_]


                        # $filenames = data_val[2]
                        with torch.cuda.amp.autocast():
                            restored = model_restoration(input_)

                            target_cpu = target.cpu().float()
                            restored_cpu = restored[0].cpu().float()
                            loss_val = criterion(target_cpu, restored_cpu)
                            mae_val.append(utils.batch_MAE(restored[0], target, False).item())
                            rmse_val.append(utils.batch_RMSE(restored[0], target, False).item())
                        epoch_loss_val += loss_val.item()
                        # print(restored.shape)
                        # visdom
                        if ii <= 3:
                            # visdom_restored = restored.cpu().numpy()
                            # visdom_target = target.cpu().numpy()
                            visdom_restored_list.append(tensor_to_heatmap(restored[0], opt.Ex_num))
                            visdom_target_list.append(tensor_to_heatmap(target, opt.Ex_num))
                        if ii == 3:
                            concatenated_visdom_restored = torch.cat(visdom_restored_list, dim=0)
                            concatenated_visdom_target = torch.cat(visdom_target_list, dim=0)
                            vizs.images(concatenated_visdom_restored,
                                        opts=dict(title='val_restored images', caption='How random.', nrow=4),
                                        win="val_restored")
                            vizs.images(concatenated_visdom_target,
                                        opts=dict(title='val_target images', caption='How random.', nrow=4),
                                        win="val_target")

                    mae_val = sum(mae_val) / len_valset
                    rmse_val = sum(rmse_val) / len_valset
                    if mae_val < best_mae:
                        best_mae = mae_val
                        best_epoch = epoch
                        best_iter = i
                        torch.save({'epoch': epoch,
                                    'state_dict': model_restoration.state_dict(),
                                    'optimizer': optimizer.state_dict()
                                    }, os.path.join(model_dir, "model_best.pth"))

                    print("[Ep %d it %d\t MAE 3D: %.4f\t] ----  [best_Ep_3D %d best_it_3D %d Best_val_MAE %.4f] " % (
                        epoch, i, mae_val, best_epoch, best_iter, best_mae))
                    print('val_RMSE on Val Set -->%.4f ' % (rmse_val))
                    with open(logname, 'a') as f:
                        f.write(
                            "[Ep %d it %d\t MAE 3D: %.4f\t] ----  [best_Ep_3D %d best_it_3D %d Best_val_MAE_3D %.4f] " \
                            % (epoch, i, mae_val, best_epoch, best_iter, best_mae) + '\n')

                # test
                with torch.no_grad():
                    model_restoration.eval()
                    mae_test = []
                    rmse_test = []
                    visdom_restored_list = []
                    visdom_target_list = []
                    for ii, data_test in enumerate((test_loader), 0):
                        target = data_test[1].to(device)
                        input_ = [img.to(device) for img in data_test[0]]
                        input_ = [img.squeeze(1) for img in input_]
                        # input_ = data_val[0].to(device)
                        # input_ = input_.squeeze(2)
                        # $filenames = data_val[2]
                        with torch.cuda.amp.autocast():
                            restored = model_restoration(input_)
                            target_cpu = target.cpu().float()
                            restored_cpu = restored[0].cpu().float()
                            loss_test = criterion(target_cpu, restored_cpu)

                        epoch_loss_test += loss_test.item()
                        # print(restored.shape)
                        # visdom
                        if ii <= 3:
                            # visdom_restored = restored.cpu().numpy()
                            # visdom_target = target.cpu().numpy()
                            visdom_restored_list.append(tensor_to_heatmap(restored[0], opt.Ex_num))
                            visdom_target_list.append(tensor_to_heatmap(target, opt.Ex_num))
                        if ii == 3:
                            concatenated_visdom_restored = torch.cat(visdom_restored_list, dim=0)
                            concatenated_visdom_target = torch.cat(visdom_target_list, dim=0)

                            vizs.images(concatenated_visdom_restored,
                                        opts=dict(title='test_restored images', caption='How random.', nrow=4),
                                        win="test_restored")
                            vizs.images(concatenated_visdom_target,
                                        opts=dict(title='test_target images', caption='How random.', nrow=4),
                                        win="test_target")
                        mae_test.append(utils.batch_MAE(restored[0], target, False).item())
                        rmse_test.append(utils.batch_RMSE(restored[0], target, False).item())
                    mae_test = sum(mae_test) / len(test_dataset)
                    rmse_test = sum(rmse_test) / len(test_dataset)
                    if mae_test < best_test_mae:
                        best_test_mae = mae_test
                        best_epoch = epoch
                        best_iter = i

                    print(
                        "[Ep %d it %d\t MAE test 3D: %.4f\t] ----  [best_Ep_3D %d best_it_3D %d Best_test_MAE %.4f] " % (
                            epoch, i, mae_test, best_epoch, best_iter, best_test_mae))
                    print('val_RMSE on Val Set -->%.4f ' % (rmse_val))
                    with open(logname, 'a') as f:
                        f.write(
                            "[Ep %d it %d\t MAE 3D: %.4f\t] ----  [best_Ep_3D %d best_it_3D %d Best_test_MAE_3D %.4f] " \
                            % (epoch, i, mae_test, best_epoch, best_iter, best_test_mae) + '\n')

                    print(' now test_MAE on Test Set -->%.4f ' % (mae_test))
                    print(' now test_RMSE on Test Set -->%.4f ' % (rmse_test))
                    model_restoration.train()
                    torch.cuda.empty_cache()
        mae_train = sum(mae_train) / len(train_dataset)
        rmse_train = sum(rmse_train) / len(train_dataset)
        print('now train_MAE on Train Set -->%.4f ' % (mae_train))
        print('now train_RMSE on Train Set -->%.4f ' % (rmse_train))
        vizs.line(
            [[epoch_loss / len(train_dataset), epoch_loss_val / len(val_dataset), epoch_loss_test / len(test_dataset)]],
            [epoch], win='loss', update='append',
            opts=dict(title='Mix_loss', legend=['train_loss', 'val_loss', 'test_loss']))

        vizs.line(
            [[epoch_loss / len(train_dataset), epoch_loss_stage0 / len(train_dataset), epoch_loss_stage1 / len(train_dataset),epoch_loss_stage2 / len(train_dataset)]],
            [epoch], win='stage_loss', update='append',
            opts=dict(title='every_stage_loss', legend=['total_loss','stage0_loss', 'stage1_loss', 'stage2_loss']))

        vizs.line(
            [[mae_train, best_mae, mae_test]],
            [epoch], win='MAE', update='append',
            opts=dict(title='MAE', legend=['train', 'val', 'test']))
        vizs.line(
            [[rmse_train, rmse_val, rmse_test]],
            [epoch], win='RMSE', update='append',
            opts=dict(title='RMSE', legend=['train', 'val', 'test']))
        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                                  (epoch_loss / len(train_dataset)),
                                                                                  scheduler.get_lr()[0]))
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tstage0_Loss: {:.4f}\tstage1_Loss: {:.4f}\tstage2_Loss: {:.4f}".format(epoch, time.time() - epoch_start_time,
                                                                                  (epoch_loss / len(train_dataset)),(epoch_loss_stage0 / len(train_dataset)),(epoch_loss_stage1 / len(train_dataset)),(epoch_loss_stage2 / len(train_dataset))))
        print("------------------------------------------------------------------")
        with open(logname, 'a') as f:
            f.write("Epoch: {}\tTime: {:.4f}\ttrain_Loss: {:.4f}\ttrain_MAE: {:.4f}\ttrain_RMSE: {:.4f}".format(epoch,
                                                                                                                time.time() - epoch_start_time,
                                                                                                                (
                                                                                                                        epoch_loss / len(
                                                                                                                    train_dataset)),
                                                                                                                mae_train,
                                                                                                                rmse_train) + '\n')
        with open(logname, 'a') as f:
            f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tstage0_Loss: {:.4f}\tstage1_Loss: {:.4f}\tstage2_Loss: {:.4f}".format(
                epoch, time.time() - epoch_start_time,
                (epoch_loss / len(train_dataset)), (epoch_loss_stage0 / len(train_dataset)),
                (epoch_loss_stage1 / len(train_dataset)), (epoch_loss_stage2 / len(train_dataset)))+ '\n')

        with open(logname, 'a') as f:
            f.write("Epoch: {}\tTime: {:.4f}\ttrain_Loss: {:.4f}\ttrain_MAE: {:.4f}\ttrain_RMSE: {:.4f}".format(epoch,
                                                                                                                time.time() - epoch_start_time,
                                                                                                                (
                                                                                                                        epoch_loss / len(
                                                                                                                    train_dataset)),
                                                                                                                mae_train,
                                                                                                                rmse_train) + '\n')
        with open(logname, 'a') as f:
            f.write("Epoch: {}\tTime: {:.4f}\ttest_Loss: {:.4f}\ttest_MAE: {:.4f}\ttest_RMSE: {:.4f}".format(epoch,
                                                                                                             time.time() - epoch_start_time,
                                                                                                             (
                                                                                                                     epoch_loss_test / len(
                                                                                                                 test_dataset)),
                                                                                                             mae_test,
                                                                                                             rmse_test) + '\n')
        with open(logname, 'a') as f:
            f.write("Epoch: {}\tTime: {:.4f}\tval_Loss: {:.4f}\tval_MAE: {:.4f}\tval_RMSE: {:.4f}".format(epoch,
                                                                                                          time.time() - epoch_start_time,
                                                                                                          (
                                                                                                                  epoch_loss_val / len(
                                                                                                              val_dataset)),
                                                                                                          mae_val,
                                                                                                          rmse_val) + '\n')
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_latest.pth"))

        if epoch % opt.checkpoint == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)))
    print("Now time is : ", datetime.datetime.now().isoformat())

