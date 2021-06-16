import torch
from torch.utils.data import DataLoader
from helper_scripts.datasets import MNIST
from helper_scripts.models import MLP
from helper_scripts.train import train

if __name__ == "__main__":
    assert torch.cuda.is_available(), "Script requires a CUDA-cabable GPU to run, found none."
    print(f"Training on {torch.cuda.device_count()} GPUs") 

    trainset, _ = MNIST("./data")
    batch_size = 128
    trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=8)

    # only additional passage w.r.t. regular model training
    net = torch.nn.DataParallel(MLP())
    # can specify also
    # DataParallel(model, device_ids=[0,1], output_device=[1])
    # this specifies that only GPUs 0 and 1 will be used
    # and that the output of the model will be found on GPU 1
    # defaults values are
    # device_ids = list(range(torch.cuda.device_count()))
    # output_device = device_ids[0]

    optimizer = torch.optim.Adam(net.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    num_epochs = 3

    train(net, optimizer, loss_fn, num_epochs, trainloader, "cuda:0")




