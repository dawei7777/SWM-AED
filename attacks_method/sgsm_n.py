import math

import torch
import torch.nn as nn

from attack import Attack


class SGSM_N(Attack):
    r"""


    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.SGSM_N(model, eps=8/255,omega=230/255, alpha=255/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255,omega=230/255, alpha=255/255):
        super().__init__("FGSM", model)
        self.eps = eps
        self.omega = omega
        self.alpha = alpha
        self.supported_mode = ["default", "targeted"]


    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.get_logits(images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]


        norm = torch.norm(grad, p=float('inf'))
        # # norm = torch.norm(grad, p=1)
        # # norm = torch.norm(grad, p=2)
        grad_inf = grad / norm


        adv_images = images + self.eps*grad_inf
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        # delta=self.eps * grad_nol
        return adv_images
