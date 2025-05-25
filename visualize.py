import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
import matplotlib.colors as mcolors
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# 定义一个转换函数，将PyTorch张量转换为PIL图片
def tensor_to_pil(image_tensor):
    # 反标准化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_tensor = image_tensor.clone()  # 避免修改原始张量
    for t, m, s in zip(image_tensor, mean, std):
        t.mul_(s).add_(m)  # 反标准化
    image_tensor = image_tensor.clamp(0, 1)  # 确保值在[0, 1]范围内
    image = transforms.ToPILImage()(image_tensor)  # 转换为PIL图片
    return image

def show_image(image_np, title='Image'):
    plt.imshow(image_np)
    plt.title(title)
    plt.axis('off')  # 不显示坐标轴
    plt.show()

def visualize_images(image, pert_image, tlabel, predicted, n=0):
    image_np = tensor_to_pil(image.cpu())
    pert_image_np = tensor_to_pil(pert_image.cpu())

    diff_image = np.abs(np.array(image_np) - np.array(pert_image_np))

    # 设置图形大小与图片大小相同
    fig1 = plt.figure(figsize=(np.array(image_np).shape[1] / 100, np.array(image_np).shape[0] / 100))
    plt.imshow(image_np)
    plt.axis('equal')  # 确保图片的宽高比保持不变
    plt.axis('off')
    fig1.savefig(r'C:\Users\13546\Pictures\SGSM\ORI\\{}_Original_image_{}.png'.format(n, tlabel.item()), dpi=100)  # 保存为PNG格式，并设置分辨率

    fig2 = plt.figure(figsize=(np.array(pert_image_np).shape[1] / 100, np.array(pert_image_np).shape[0] / 100))
    plt.imshow(pert_image_np)
    plt.axis('equal')  # 确保图片的宽高比保持不变
    plt.axis('off')
    fig2.savefig(r'C:\Users\13546\Pictures\SGSM\PER\\{}_Perturbed_image_{}.png'.format(n, predicted.item()), dpi=100)  # 保存为PNG格式，并设置分辨率

    fig3 = plt.figure(figsize=(diff_image.shape[1] / 100, diff_image.shape[0] / 100))
    plt.imshow(diff_image, cmap='hot', norm=mcolors.PowerNorm(gamma=0.01))
    plt.axis('equal')  # 确保图片的宽高比保持不变
    plt.axis('off')
    fig3.savefig(r'C:\Users\13546\Pictures\SGSM\DIF\\{}_Difference_image.png'.format(n), dpi=100)  # 保存为PNG格式，并设置分辨率
    plt.show()