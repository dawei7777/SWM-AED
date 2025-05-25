import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
import matplotlib.colors as mcolors
from torchvision import transforms
import torch
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

def tensorx_to_pil(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    image_tensor = image_tensor.clone()  # 避免修改原始张量
    image_tensor = image_tensor * std + mean  # 反标准化
    image_tensor = image_tensor.clamp(0, 1)  # 确保值在[0, 1]范围内
    image = transforms.ToPILImage()(image_tensor)  # 转换为PIL图片
    return image

# 反标准化函数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).to(device)
def inverse_normalize(tensor, mean, std):
    for c in range(tensor.size(1)):  # 遍历所有通道
        tensor[:, c, :, :] = tensor[:, c, :, :] * std[c] + mean[c]
    return tensor.clamp(0, 1)  # 确保张量值在[0, 1]范围内



def visualize_images(image, delta, tlabel, predicted, n=0):
    original_image = inverse_normalize(image.clone(), mean, std)

    perturbed_image = original_image + delta
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    plt.subplot(1, 3, 1)
    plt.imshow(original_image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    plt.title('Original Image {}'.format(tlabel))

    plt.subplot(1, 3, 2)
    plt.imshow(perturbed_image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    plt.title('Perturbed Image {}'.format(predicted))

    plt.subplot(1, 3, 3)
    diff = torch.abs(original_image - perturbed_image)
    plt.imshow(diff.squeeze().permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
    plt.title('Difference')
    plt.show()



def visualize_images_x(image, pert_image, tlabel, predicted, n=0):
    image_np = tensorx_to_pil(image.cpu())

    pert_image_np = tensorx_to_pil(pert_image.cpu())

    diff_image = np.abs(np.array(image_np) - np.array(pert_image_np))

    # 创建一个包含三张子图的图形
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 在第一张子图上绘制原始图像
    axs[0].imshow(image_np)
    axs[0].set_title('Original Image {}'.format(tlabel.item()), fontsize=12)
    axs[0].axis('off')

    # 在第二张子图上绘制扰动后的图像
    axs[1].imshow(pert_image_np)
    axs[1].set_title('Perturbed Image {}'.format(predicted.item()), fontsize=12)
    axs[1].axis('off')

    # 在第三张子图上绘制差异图像
    axs[2].imshow(diff_image, cmap='hot', norm=mcolors.PowerNorm(gamma=0.01))
    axs[2].set_title('Difference Image')
    axs[2].axis('off')

    # 调整布局，确保子图不会重叠
    plt.tight_layout()
    # 显示图形
    plt.show()