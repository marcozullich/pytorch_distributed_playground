import torch
from torch.nn.modules import loss
from torch.utils.data import DataLoader

import argparse

from helper_scripts.datasets import MNIST
from helper_scripts.models import MLP
from helper_scripts.train import train

class MLPParallel(MLP):
    def __init__(self, devices=["cuda:0", "cuda:1"], **kwargs):
        assert torch.cuda.device_count() >= 2, f"Expected machine to have at least 2 CUDA-capable GPUs, found {torch.cuda.device_count()}"
        super().__init__(**kwargs)
        self.devices = devices
        self.chunk1 = self.chunk1.to(self.devices[0])
        self.chunk2 = self.chunk2.to(self.devices[1])
    
    def forward(self, data, print_shape=False):
        if print_shape:
            print(f"\t\tData shape in forward: {data.shape}")
        out = self.chunk1(data).to(self.devices[1])
        return self.chunk2(out)

class MLPParallelPipelined(MLPParallel):
    def __init__(self, split_size=20, **kwargs):
        super().__init__(**kwargs)
        self.split_size = split_size
    
    def forward(self, data, print_shape=False):
        if print_shape:
            print(f"\t\tData shape in forward: {data.shape}")
        splits = iter(data.split(self.split_size, dim=0))

        split = next(splits)
        intermediate_eval = self.chunk1(split).to(self.devices[1])

        outputs = []
        for split in splits:
            out = self.chunk2(intermediate_eval)
            outputs.append(out)

            intermediate_eval = self.chunk2(split).to(self.devices[1])
        
        return torch.cat(outputs, dim=0)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipelined", default=False, action="store_true")
    args = parser.parse_args()

    print(f"Training in model parallel{' pipelined' if args.pipelined else ''} mode")

    trainset, _ = MNIST("./data")
    batch_size = 128
    trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=8)

    net = MLPParallelPipelined() if args.pipelined else MLPParallel()

    optimizer = torch.optim.Adam(net.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    num_epochs = 3

    train(net, optimizer, loss_fn, num_epochs, trainloader, "cuda:0")


