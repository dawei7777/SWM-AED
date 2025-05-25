import argparse
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
# from mnist_net import mnist_net
from visualize_save import visualize_images,visualize_images_x
from utils.readData import read_dataset
from utils.ResNet import ResNet18
from PIL import Image
import matplotlib.pyplot as plt
import csv
import torchattacks
from YOAO import yoao

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--data-dir', default='../mnist-data', type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--attack', default='fgsm', type=str, choices=['none', 'pgd', 'fgsm','sgsm','fgsm(x)'])
    parser.add_argument('--epsilon', default=0.001, type=float)
    parser.add_argument('--alpha', default=0.375, type=float)
    parser.add_argument('--attack-iters', default=40, type=int)
    parser.add_argument('--fname', default='mnist_model_none', type=str)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()

from PIL import Image
import torch
from torchvision import transforms



# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def apply_mask(image, mask_size, position, color=(0, 1, 0)):
    """
    在图像上应用一个指定颜色的mask遮挡。

    :param image: 输入图像张量，形状应为[1, C, H, W]。
    :param mask_size: 遮挡的大小，例如(3, 3)。
    :param position: 遮挡的起始位置，例如(12, 5)。
    :param color: 遮挡的颜色，RGB值，例如(0, 1, 0)表示绿色。
    :return: 应用遮挡后的图像张量。
    """
    # 创建一个指定颜色的mask张量
    # mask = torch.tensor(color, dtype=torch.float32).view(3, 1, 1).repeat(1, mask_size[0], mask_size[1])
    mask = torch.tensor(color, dtype=image.dtype).view(3, 1, 1).repeat(1, mask_size[0], mask_size[1])
    # 将mask放置到图像的指定位置
    image[:, :, position[0]:position[0] + mask_size[0], position[1]:position[1] + mask_size[1]] = mask
    return image


def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_class = 10
    model = ResNet18()  # 得到预训练模型
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(512, n_class)  # 将最后的全连接层修改
    # 载入权重
    model.load_state_dict(torch.load('checkpoint/resnet18_cifar10.pt'))
    model = model.to(device)
    model.eval()  # 验证模型

    criterion = nn.CrossEntropyLoss()

    batch_size = 1
    train_loader, valid_loader, test_loader = read_dataset(batch_size=batch_size, pic_path='dataset')
    # 定义攻击器
    attack = torchattacks.JSMA(model, theta=1.0, gamma=0.1)
    # attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
    # attack = torchattacks.DeepFool(model, steps=10, overshoot=0.02)
    # attack = torchattacks.FGSM(model, eps=8/255)
    # attack = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
    # attack = torchattacks.FFGSM(model, eps=8/255, alpha=10/255)
    # attack = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
    # attack = torchattacks.SparseFool(model, steps=5, lam=3, overshoot=0.02)
    # attack = torchattacks.OnePixel(model, pixels=1, steps=10, popsize=10, inf_batch=128)
    # attack = torchattacks.Pixle(model, x_dimensions=(0.1, 0.2), restarts=10, max_iterations=10)
    # attack = torchattacks.PIFGSMPP(model,  num_iter_set=10)

    count = 0
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        # 模拟一个输入
        input_image = X.requires_grad_(True)

        # 获取模型输出
        output = model(input_image)
        loss = output.mean()  # 这里使用均值作为损失，实际使用时可能需要根据任务调整
        loss.backward()
        confidence_array = np.zeros((26, 26))
        confidence_array_ori = np.zeros((26, 26))
        smc_tol=0
        smc_tol_ori=0
        n=0

        for i in range(0,26,9):
            for j in range(0,26,9):
                # _, _, _, _, perturbed_image = yoao(input_image.detach(), y, model, 10,overshoot=0.02, max_iter=1)
                perturbed_image = attack(input_image, y)  # 确保输入是正确的形状

                perturbed_image = apply_mask(perturbed_image, (9, 9), (i, j), color=(0, 1, 0))
                image_copy = input_image.clone()
                ori_image = apply_mask(image_copy, (9, 9), (i, j), color=(0, 1, 0))
                output = model(ori_image)
                output_per = model(perturbed_image)
                # 计算置信度分数
                confidence = F.softmax(output, dim=1).max().item()
                confidence_per = F.softmax(output_per, dim=1).max().item()
                conf = F.softmax(output, dim=1)
                conf_per = F.softmax(output_per, dim=1)
                confidence_array_ori[i,j]=confidence
                confidence_array[i, j] = confidence_per
                #  SMC
                smc = -torch.sum(conf * torch.log2(conf))
                smc_per = -torch.sum(conf_per * torch.log2(conf_per))
                smc_tol_ori+=smc
                smc_tol+=smc_per
                n=n+1
                # 可视化
        smc_avr=smc_tol/n
        smc_array_ori_avr=smc_tol_ori/n
        print(smc_array_ori_avr)
        print(smc_avr)


        plt.figure(figsize=(12, 6))
        plt.subplot(1, 4, 1)
        class_labels_en = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        plt.imshow(X.squeeze().permute(1, 2, 0).detach().cpu().numpy())  # 转换为可显示的格式
        plt.title('Original Image {} \n (Confidence: {:.2f})'.format(class_labels_en[output.argmax().item()], confidence))

        plt.subplot(1, 4, 2)
        plt.imshow(perturbed_image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
        plt.title('Perturbed Image {} \n (Confidence: {:.2f})'.format(class_labels_en[output_per.argmax().item()], confidence_per))

        plt.subplot(1, 4, 3)
        plt.imshow(confidence_array_ori, cmap='hot')
        plt.title('Confidence Array_ori \n (SMCE_ori: {:.2f})'.format(smc_array_ori_avr.item()))
        #plt.savefig(r'C:\Users\13546\Pictures\SGSM\MASK\\{}_Perturbed_image_{}.png'.format(i,j), dpi=100)  # 保存为PNG格式，并设置分辨率

        plt.subplot(1, 4, 4)
        plt.imshow(confidence_array, cmap='hot')
        plt.colorbar()
        plt.title('Confidence Array_per \n (SMCE_per: {:.2f})'.format(smc_avr.item()))
        count = count + 1
        # plt.savefig(r'C:\Users\13546\Pictures\SGSM\SWM-AED\\{}_SWM-AED_{}.png'.format(i, count), dpi=100)  # 保存为PNG格式，并设置分辨率

        plt.show()



if __name__ == "__main__":
    main()