import random

import imgaug.augmenters as iaa
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


class AdditiveGaussianNoise:
    def __init__(self, scale=(10, 30)):
        self.aug = iaa.AdditiveGaussianNoise(scale=scale)

    def __call__(self, image):
        """
        Apply Gaussian noise to a PyTorch image tensor.

        Parameters:
            image (torch.Tensor): The input image tensor (C, H, W) with values in [0, 255].

        Returns:
            torch.Tensor: The noisy image tensor (C, H, W).
        """
        # Convert PyTorch tensor to numpy array
        image = image.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)

        # Apply imgaug augmentations
        image = self.aug(image=image)

        # Convert back to PyTorch tensor
        return torch.tensor(image).permute(2, 0, 1)  # Change back to (C, H, W)


class RandomShift(object):
    def __init__(self, max_shift):
        self.max_shift = max_shift

    def __call__(self, image):
        # Randomly choose shifts in both directions
        shift_x = random.randint(-self.max_shift, self.max_shift)
        shift_y = random.randint(-self.max_shift, self.max_shift)

        # Apply the affine transformation
        # No rotation, no scaling, no shear, and the random translation
        shifted_image = F.affine(image, angle=0, translate=(shift_x, shift_y), scale=1, shear=0)

        return shifted_image

    def __repr__(self):
        return f"{self.__class__.__name__}(max_shift={self.max_shift})"


class KernelDS(Dataset):
    MATCH = 1
    NOMATCH = 0

    def __init__(self, file, ds_size, base_transforms=None, max_shift_pixels=100):
        self.files = [file] * ds_size
        self.max_shift_pixels = max_shift_pixels

        if not base_transforms:
            base_transforms = []
        self.match_transforms = transforms.Compose(
            base_transforms + [
                # transforms.RandomRotation(degrees=180),  # Random rotation up to 30 degrees
                transforms.ToTensor(),
                # AdditiveGaussianNoise(),  # Add Gaussian noise
            ])
        self.nomatch_transforms = transforms.Compose(base_transforms +
                                                     [RandomShift(max_shift=self.max_shift_pixels),
                                                      #                                              transforms.RandomChoice(
                                                      # [, ]),
                                                      transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path)

        if (idx % 2) == 0:
            label = self.NOMATCH
            transforms = self.nomatch_transforms

        else:
            label = self.MATCH
            transforms = self.match_transforms

        image = transforms(image)

        return image, torch.tensor(label, dtype=torch.float32)

    def show_sample(self):
        # show sample
        idx = random.randint(0, len(self))
        print(self[idx][1])
        to_img = transforms.ToPILImage()
        return to_img(self[idx][0])
