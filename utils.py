import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg[:4])).eval()
        self.slice2 = nn.Sequential(*list(vgg[4:9])).eval()
        self.slice3 = nn.Sequential(*list(vgg[9:16])).eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        x = F.interpolate(x, mode='bilinear', size=(224, 224), align_corners=False)
        y = F.interpolate(y, mode='bilinear', size=(224, 224), align_corners=False)
        x_relu1_2, x_relu2_2, x_relu3_3 = self.slice1(x), self.slice2(x), self.slice3(x)
        y_relu1_2, y_relu2_2, y_relu3_3 = self.slice1(y), self.slice2(y), self.slice3(y)
        loss = F.mse_loss(x_relu1_2, y_relu1_2) + F.mse_loss(x_relu2_2, y_relu2_2) + F.mse_loss(x_relu3_3, y_relu3_3)
        return loss

class AverageMeter(object):
    """ Function to Compute and store the average and current value
    """

    def __init__(self):

        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_psnr(img1, img2):
    """ Function to calculate the Peak Signal to Noise Ratio (PSNR)

    :param img1: model output Isr image
    :param img2: ground truth Ihr image
    :return: Peak Signal to Noise Ratio between two images
    """

    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))
