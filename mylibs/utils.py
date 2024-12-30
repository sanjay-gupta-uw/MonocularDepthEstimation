import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Based off of the following implementation
# https://github.com/OniroAI/MonoDepth-PyTorch/blob/master/loss.py

class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1 ):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        padding = (self.kernel_size - 1) // 2
        x = self.convolution(F.pad(x, (padding, padding, padding, padding)))
        x = self.norm(x)
        return F.elu(x, inplace=True)

class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(in_channels, out_channels, kernel_size, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)

class max_pool(nn.Module):
    def __init__(self, kernel_size):
        super(max_pool, self).__init__()
        self.kernel_size = kernel_size
        
    def forward(self, x):
        padding = (self.kernel_size - 1) // 2
        return F.max_pool2d(F.pad(x, (padding, padding, padding, padding)), self.kernel_size, stride=2)
    
class resconv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(resconv, self).__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = conv(in_channels, out_channels, 3, stride)
        self.conv2 = conv(out_channels, out_channels, 3, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        shortcut = self.conv3(x)

        return F.elu(self.normalize(x_out + shortcut), inplace=True)

def resblock(in_channels, out_channels, num_blocks, stride):
    layers = [resconv(in_channels, out_channels, stride)]
    for i in range(1, num_blocks):
        layers.append(resconv(out_channels, out_channels, 1))
    return nn.Sequential(*layers)


class get_disparity(nn.Module):
    def __init__(self, in_channels):
        super(get_disparity, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1)
        self.norm = nn.BatchNorm2d(2)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        padding = 1
        x = self.conv1(F.pad(x, (padding, padding, padding, padding)))
        x = self.norm(x)
        return self.activation(x) * 0.3
    
# post-process disparity maps function to convert to depth maps:
def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


class MDELoss(nn.modules.Module):
    def __init__(self, n=4, SSIM_w=0.85, disp_gradient_w=1.0, lr_w=1.0):
        super(MDELoss, self).__init__()
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.num_images = n # num images in the pyramid

    def multiresolution_images(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.interpolate(img,
                               size=[nh, nw], mode='bilinear',
                               align_corners=True))
        return scaled_imgs

    def gradient_x(self, img):
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear', padding_mode='zeros') # In grid_sample coordinates are assumed to be between -1 and 1

        return output

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x **2
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y **2
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - (mu_x * mu_y)

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x **2 + mu_y **2 + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, multires):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in multires]
        image_gradients_y = [self.gradient_y(img) for img in multires]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

        smoothness = [
            torch.abs(disp_gradients_x[i] * weights_x[i]) +
            torch.abs(disp_gradients_y[i] * weights_y[i])
            for i in range(self.num_images)
        ]
        return smoothness

    def forward(self, input, target):
        """
        Args:
            input [disp1, disp2, disp3, disp4]
            target [left, right]

        Return:
            (float): The loss
        """
        left, right = target
        left_multires = self.multiresolution_images(left, self.num_images)
        right_multires = self.multiresolution_images(right, self.num_images)

        # Prepare disparities
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in input]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in input]

        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est
        # Generate images
        left_est = [self.generate_image_left(right_multires[i], disp_left_est[i])
                          for i in range(self.num_images)]
        right_est = [self.generate_image_right(left_multires[i], disp_right_est[i])
                          for i in range(self.num_images)]
        self.left_est = left_est
        self.right_est = right_est
        
        # L-R Consistency
        right_left_disp = [self.generate_image_left(disp_right_est[i], disp_left_est[i])
                           for i in range(self.num_images)]
        left_right_disp = [self.generate_image_right(disp_left_est[i], disp_right_est[i]) 
                           for i in range(self.num_images)]

        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_left_est,
                                                    left_multires)
        disp_right_smoothness = self.disp_smoothness(disp_right_est,
                                                     right_multires)

        # L1
        l1_left = [torch.mean(torch.abs(left_est[i] - left_multires[i]))
                   for i in range(self.num_images)]
        l1_right = [torch.mean(torch.abs(right_est[i] - right_multires[i])) 
                    for i in range(self.num_images)]

        # SSIM
        ssim_left = [torch.mean(self.SSIM(left_est[i], left_multires[i])) 
                     for i in range(self.num_images)]
        ssim_right = [torch.mean(self.SSIM(right_est[i], right_multires[i]))
                      for i in range(self.num_images)]

        image_loss_left = [self.SSIM_w * ssim_left[i] + (1 - self.SSIM_w) * l1_left[i]
                           for i in range(self.num_images)]
        image_loss_right = [self.SSIM_w * ssim_right[i] + (1 - self.SSIM_w) * l1_right[i]
                            for i in range(self.num_images)]
        image_loss = sum(image_loss_left + image_loss_right)

        # L-R Consistency
        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i] - disp_left_est[i]))
                        for i in range(self.num_images)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i] - disp_right_est[i]))
                         for i in range(self.num_images)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_loss = [torch.mean(torch.abs(disp_left_smoothness[i])) / 2 ** i
                          for i in range(self.num_images)]
        disp_right_loss = [torch.mean(torch.abs(disp_right_smoothness[i])) / 2 ** i
                           for i in range(self.num_images)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        loss = image_loss + self.disp_gradient_w * disp_gradient_loss + self.lr_w * lr_loss
        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss
        return loss