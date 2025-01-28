
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def compute_losses(image1, image2, flow_pred, image2_pred, photometric_type="l1", smoothness_order=1):
    # photometric
    l_photometric = _compute_photometric_loss(image2_pred, image2, photometric_type)

    # smoothness regularization
    l_smooth_reg = _compute_smoothness_reg(flow_pred, image1, order=smoothness_order)

    # self-supervision (teacher-student)

    losses = {"photometric": l_photometric, "smooth_reg": l_smooth_reg}
    return losses


def _compute_photometric_loss(I_pred, I_true, key="l1_robust", photo_loss_delta=0.4):
    loss = None
    if key == "l1":
        loss = torch.abs(I_pred - I_true + 1e-6)
    elif key == "l1_robust":
        loss = (torch.abs(I_pred - I_true) + 0.1).pow(photo_loss_delta)
    elif key == "carbonnier":
        loss = ((I_pred - I_true)**2 + 1e-6).pow(photo_loss_delta)
    elif key == "ssim":
        loss = _compute_weighted_ssim(I_pred, I_true, weight=torch.ones_like(I_pred).to(I_pred.device))
    else:
        raise NotImplementedError("Unsupported photometric loss key: {}".format(key))
    
    return loss.mean()

def _compute_smoothness_reg(flow, image, alpha=10, order=1):
    if order == 1:
        img_dx, img_dy = _gradient(image)
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

        dx, dy = _gradient(flow)

        loss_x = weights_x * dx.abs() / 2.
        loss_y = weights_y * dy.abs() / 2

        return loss_x.mean() / 2. + loss_y.mean() / 2.
    elif order == 2:
        img_dx, img_dy = _gradient(image)
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

        dx, dy = _gradient(flow)
        dx2, dxdy = _gradient(dx)
        dydx, dy2 = _gradient(dy)

        loss_x = weights_x[:, :, :, 1:] * dx2.abs()
        loss_y = weights_y[:, :, 1:, :] * dy2.abs()

        return loss_x.mean() / 2. + loss_y.mean() / 2.
    else:
        raise NotImplementedError("Unsupported smoothing regularization order")

def _gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy

def _compute_weighted_ssim(x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
    """Computes a weighted structured image similarity measure.
    Args:
        x: a batch of images, of shape [B, C, H, W].
        y: a batch of images, of shape [B, C, H, W].
        weight: shape [B, 1, H, W], representing the weight of each
        pixel in both images when we come to calculate moments (means and
        correlations). values are in [0,1]
        c1: A floating point number, regularizes division by zero of the means.
        c2: A floating point number, regularizes division by zero of the second
        moments.
        weight_epsilon: A floating point number, used to regularize division by the
        weight.

    Returns:
        A tuple of two pytorch Tensors. First, of shape [B, C, H-2, W-2], is scalar
        similarity loss per pixel per channel, and the second, of shape
        [B, 1, H-2. W-2], is the average pooled `weight`. It is needed so that we
        know how much to weigh each pixel in the first tensor. For example, if
        `'weight` was very small in some area of the images, the first tensor will
        still assign a loss to these pixels, but we shouldn't take the result too
        seriously.
    """

    def _avg_pool3x3(x):
        return F.avg_pool2d(x, (3, 3), (1, 1))

    if c1 == float('inf') and c2 == float('inf'):
        raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                            'likely unintended.')
    average_pooled_weight = _avg_pool3x3(weight)
    weight_plus_epsilon = weight + weight_epsilon
    inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

    def weighted_avg_pool3x3(z):
        wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
        return wighted_avg * inverse_average_pooled_weight

    mu_x = weighted_avg_pool3x3(x)
    mu_y = weighted_avg_pool3x3(y)
    sigma_x = weighted_avg_pool3x3(x ** 2) - mu_x ** 2
    sigma_y = weighted_avg_pool3x3(y ** 2) - mu_y ** 2
    sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
    if c1 == float('inf'):
        ssim_n = (2 * sigma_xy + c2)
        ssim_d = (sigma_x + sigma_y + c2)
    elif c2 == float('inf'):
        ssim_n = 2 * mu_x * mu_y + c1
        ssim_d = mu_x ** 2 + mu_y ** 2 + c1
    else:
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    result = ssim_n / ssim_d
    return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight