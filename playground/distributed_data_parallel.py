import torch
import os
import argparse

from helper_scripts.datasets import MNIST
from helper_scripts.models import MLP
from helper_scripts.train import train

def prepare_train(gpu, args):
    rank = args.gpus * args.nr + gpu
    torch.distirbuted.init_process_group(
        backend = "nccl",
        init_method = "enc://",
        world_size = args.world_size,
        rank = rank
    )

    device = f"cuda:{gpu}"
    net = MLP().to(device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu])

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters())
    trainset, _ = MNIST()

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replices=args.world_size, rank=rank)
    # shuffle is False bc. DistributedSampler already shuffles
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=train_sampler)

    train(net, optimizer, loss_fn, args.num_epochs, trainloader, device, cuda_non_blocking=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--epochs', default=3, type=int, help='number of total epochs to run')
    parser.add_argument("--batch_size", default=128, type=int, help="batch size for trainset")
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes

    os.environ["MASTER_ADDR"] = "10.57.23.164"
    os.environ["MASTER_PORT"] = "8888"

    torch.multiprocessing.spawn(prepare_train, nprocs=args.gpus, args = (args,))

