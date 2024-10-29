import torch.nn as nn


class TemplateMatcher3DKernel(nn.Module):
    def __init__(self, h, w, d=3):
        # use d=3 for colored images and 1 for black and white
        super(TemplateMatcher3DKernel, self).__init__()
        self.h, self.w, self.d = h, w, d
        self.kernel = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(d, h, w))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.kernel(x)
        x = self.sigmoid(x).flatten()
        return x

    def convolve(self, image):
        padding_d = 0
        padding_h = (self.kernel.kernel_size[1] - 1) // 2  # Same padding for height
        padding_w = (self.kernel.kernel_size[2] - 1) // 2  # Same padding for width
        self.kernel.padding = (padding_d, padding_h, padding_w)
        x = self.kernel(image)
        return self.sigmoid(x)
