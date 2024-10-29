import argparse

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from kernel import TemplateMatcher3DKernel
from utils import load_dimensions_from_yaml, save_image_with_suffix

transforms = transforms.ToTensor()


def find_matches(model, img_path, thresh=0.9, device="cuda"):
    img = Image.open(img_path)
    transformed_img = transforms(img)
    transformed_img = transformed_img.unsqueeze(0)
    transformed_img = transformed_img.to(device)
    result = model.convolve(transformed_img)
    matches = (result >= thresh).nonzero()
    return matches, img


def plot(matches, kernel_w, kernel_h, img, color='#FF0000'):
    draw = ImageDraw.Draw(img)
    for _, _, y, x in matches.tolist():
        x1 = int(x - kernel_w / 2)
        y1 = int(y - kernel_h / 2)
        x2 = int(x + kernel_w / 2)
        y2 = int(y + kernel_h / 2)
        draw.rectangle([x1, y1, x2, y2], outline=color)

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detects a Template using trained kernel")
    parser.add_argument('--image_path', type=str, help='Image to do lookup on', required=True)
    parser.add_argument('--thresh', type=float, default=0.9, help='match threshold')
    parser.add_argument('--device', type=str, default='cuda', help='device use to train model')
    parser.add_argument('--weights_path', type=str, default='template_matcher.pth', help='path to model weights.pth')
    parser.add_argument('--kernelconfig_path', type=str, default='kernelconfig.yaml',
                        help='Path to save the trained kernel config')
    parser.add_argument('--plot', type=bool, default=True, help='Plot the result')
    args = parser.parse_args()

    h, w, d = load_dimensions_from_yaml(args.kernelconfig_path)

    model = TemplateMatcher3DKernel(h, w, d)
    state_dict = torch.load(args.weights_path)
    model.load_state_dict(state_dict)
    model.to(args.device)
    matches, img = find_matches(model, args.image_path, thresh=args.thresh, device=args.device)
    print(matches)
    if args.plot:
        img = plot(matches, h, w, img)
        save_image_with_suffix(img, args.image_path)
