import os

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from solution.model import loadMyModel
from solution.data import load_datasets, INPUT_IMG_SIZE

import matplotlib
import matplotlib.pyplot as plt

from PIL import Image

PATH = os.path.join(os.getcwd(), 'fruits-360/Training/Pear Kaiser/72_100.jpg')
CLASSNAME = 'Pear Kaiser'
# PATH = os.path.join(os.getcwd(), 'fruits-360/Training/Pineapple Mini/178_100.jpg')
# CLASSNAME = 'Pineapple Mini'
IMAGE_DEST_PATH = os.path.join(os.getcwd(), "occluded" + CLASSNAME + ".png")

OCCLUSION_SIZE = 8

OCCLUSION_SKIP = 1

image_datasets, dataloaders, dataset_sizes, class_names = load_datasets()
net, device = loadMyModel(class_names)


def occludedImage(img, x, y, width = OCCLUSION_SIZE):
    img = img.clone()
    width = OCCLUSION_SIZE
    for channelId in range(len(img)):
        for i in range(max(0, x-width), min(x+width, INPUT_IMG_SIZE)):
            for j in range(max(0, y-width), min(y+width, INPUT_IMG_SIZE)):
                img[channelId][i][j] = 0.5
    return img


def randomImage():
    # # RANDOM
    # data, targets = next(iter(dataloaders['Test']))
    # return data[0], targets[0]

    # FROMPATH
    data = Image.open(PATH)
    data = transforms.ToTensor()(data)
    target = class_names.index(CLASSNAME)
    target = torch.tensor(target, dtype=torch.long)
    return data, target

def main():
    img, tgt = randomImage()

    img, tgt = img.to(device), tgt.to(device)

    crit = nn.CrossEntropyLoss()
    losses = torch.zeros(INPUT_IMG_SIZE, INPUT_IMG_SIZE)
    with torch.no_grad():
        for x in range(0, int(INPUT_IMG_SIZE / OCCLUSION_SKIP)):
            print('col', x)
            for y in range(0, int(INPUT_IMG_SIZE / OCCLUSION_SKIP)):
                ocl = occludedImage(img, x, y)

                output = net(ocl.unsqueeze(0)) # emulate batch with size 1
                losses[x][y] = crit(output, tgt.unsqueeze(0)).item()

    plt.imshow(losses.cpu(), cmap='hot', interpolation='nearest')
    plt.savefig(IMAGE_DEST_PATH)
main()