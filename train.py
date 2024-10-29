import argparse
import copy
import time

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from data import KernelDS
from kernel import TemplateMatcher3DKernel
from utils import save_dimensions_to_yaml

def get_image_shape(image_path):
    image = cv2.imread(image_path)

    # Get image dimensions and number of channels
    return image.shape


def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    since = time.time()

    val_acc_history = []
    val_loss = []
    train_loss = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss

                    outputs = model(inputs)
                    preds = (outputs > 0.9).to(torch.float32)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_loss, val_loss


def train(args):
    h, w, n_channels = get_image_shape(args.template_image)
    train_ds = KernelDS(args.template_image, 300)
    test_ds = KernelDS(args.template_image, 60)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=4)
    data_loaders = {'train': train_dl, 'val': val_dl}
    model = TemplateMatcher3DKernel(h, w, n_channels)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=0.005, momentum=0.9)
    criterion = nn.MSELoss()
    model = model.to(args.device)

    model, val_acc_history, train_loss, val_loss = train_model(model, data_loaders, criterion, optimizer,
                                                               args.epochs, args.device)
    torch.save(model.state_dict(), args.save_path)
    save_dimensions_to_yaml(model.h, model.w, model.d, args.kernelconfig_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Template matching kernel")
    parser.add_argument('--template_image', type=str, help='Image path for the template', required=True)
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate for optimizer')
    parser.add_argument('--save_path', type=str, default='template_matcher.pth', help='Path to save the trained model')
    parser.add_argument('--kernelconfig_path', type=str, default='kernelconfig.yaml', help='Path to save the trained kernel config')
    parser.add_argument('--device', type=str, default='cuda', help='device use to train model')
    args = parser.parse_args()
    train(args)
