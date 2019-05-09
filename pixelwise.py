import os

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from PIL import Image

import matplotlib
import matplotlib.pyplot as plt

from solution.model import loadMyModel
from solution.data import load_datasets, INPUT_IMG_SIZE

PATH = os.path.join(os.getcwd(), 'fruits-360/Training/Pear Kaiser/72_100.jpg')
CLASSNAME = 'Pear Kaiser'
# PATH = os.path.join(os.getcwd(), 'fruits-360/Training/Pineapple Mini/178_100.jpg')
# CLASSNAME = 'Pineapple Mini'
IMAGE_DEST_PATH = os.path.join(os.getcwd(), "pixelwise" + CLASSNAME + ".png")


image_datasets, dataloaders, dataset_sizes, class_names = load_datasets()
net, device = loadMyModel(class_names)


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
    net.train()
    img, tgt = randomImage()

    img, tgt = img.to(device), tgt.to(device)
    batch = img.unsqueeze(0)
    batch = torch.autograd.Variable(batch, requires_grad=True)

    crit = nn.CrossEntropyLoss()
    output = net(batch) # emulate batch with size 1
    
    loss = crit(output, tgt.unsqueeze(0))
    loss.backward(retain_graph=True)
    print(batch.grad)



    plt.imshow(torch.sum(batch.grad[0], 0).cpu(), cmap='hot', interpolation='nearest')
    plt.savefig(IMAGE_DEST_PATH)
main()