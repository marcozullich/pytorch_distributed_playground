from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import datasets
import os

def MNIST(data_root):
    os.makedirs(data_root, exist_ok=True)

    transforms = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])

    trainset = datasets.MNIST(data_root, train=True, transform=transforms, download=True)
    testset = datasets.MNIST(data_root, train=False, transform=transforms, download=True)

    return trainset, testset