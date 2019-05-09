import os

import torch

import torchvision
from torchvision import datasets, transforms

DATASET_PATH = os.path.join(os.getcwd(), 'fruits-360')
ROOT_SUFFIXES = ["Training", "Test"]

BATCH_SIZE = 92
NUM_WORKERS = 32

INPUT_IMG_SIZE = 100 # images are scaled to a square with this edge length (preventing a case where a image has different dimensions)

def load_datasets():
    transformations = transforms.Compose([
        transforms.Resize((INPUT_IMG_SIZE, INPUT_IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATASET_PATH, x),
                                              transformations)
                      for x in ROOT_SUFFIXES}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                 shuffle=True, num_workers=NUM_WORKERS)
                    for x in ROOT_SUFFIXES}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ROOT_SUFFIXES}
    class_names = image_datasets['Training'].classes

    return image_datasets, dataloaders, dataset_sizes, class_names
