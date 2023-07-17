import torch.nn.functional as F
import torch
import copy
from tqdm import trange
import math
import torch.nn as nn
import random


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=7, max_sigma=1, sample_sigma=True):
        super(GaussianBlur, self).__init__()

        self.kernel_size = kernel_size
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        self.xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        self.mean = (kernel_size - 1) / 2.
        self.max_sigma = max_sigma
        self.kernel_size = kernel_size
        self.sample_sigma = sample_sigma

    def random_kernel(self):
        if self.sample_sigma:
            sigma = random.random() * self.max_sigma + 0.01
        else:
            sigma = self.max_sigma
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(-torch.sum((self.xy_grid - self.mean) ** 2., dim=-1) / (2 * variance))
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = (gaussian_kernel / torch.sum(gaussian_kernel)).reshape(1, 1, self.kernel_size, self.kernel_size)
        return gaussian_kernel.expand(3, -1, -1, -1)

    def forward(self, x):
        gaussian_kernel = self.random_kernel()
        return F.conv2d(x, gaussian_kernel.to(x.device), bias=None, padding=self.kernel_size//2, stride=1, groups=3)


def noise_from_x0(curr_img, img_pred, alpha):
    return (curr_img - alpha.sqrt() * img_pred)/((1 - alpha).sqrt() + 1e-4)


def cosine_alphas_bar(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_bar = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    return alphas_bar


def image_cold_diffuse_simple(diffusion_model, batch_size, total_steps, device, image_size=32, no_p_bar=True, noise_sigma=1):
    diffusion_model.eval()

    random_image_sample = noise_sigma * torch.randn(batch_size, 3, image_size, image_size, device=device)
    sample_in = copy.deepcopy(random_image_sample)

    alphas = torch.flip(cosine_alphas_bar(total_steps), (0,)).to(device)

    for i in trange(total_steps-1, disable=no_p_bar):
        index = i * torch.ones(batch_size, device=device)
        img_output = diffusion_model(sample_in, index)["image_out"]

        noise = noise_from_x0(sample_in, img_output, alphas[i])
        x0 = img_output

        rep1 = alphas[i].sqrt() * x0 + (1 - alphas[i]).sqrt() * noise
        rep2 = alphas[i + 1].sqrt() * x0 + (1 - alphas[i + 1]).sqrt() * noise

        sample_in += rep2 - rep1

    index = (total_steps - 1) * torch.ones(batch_size, device=device)
    img_output = diffusion_model(sample_in, index)["image_out"]

    return torch.clamp(img_output, -1, 1), torch.clamp(random_image_sample, -1, 1)
