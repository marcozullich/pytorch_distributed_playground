'''
from https://towardsdatascience.com/how-to-convert-a-pytorch-dataparallel-project-to-use-distributeddataparallel-b84632eed0f6
'''

import torch
import os
import argparse
import socket
import logging

from datetime import timedelta

from detectron2.utils import comm

from helper_scripts.datasets import MNIST
from helper_scripts.models import MLP
from helper_scripts.train import AverageMeter, accuracy


DEFAULT_TIMEOUT = timedelta(minutes=5)

def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def launch(
    main_func,
    gpus_per_node,
    nodes=1,
    rank=0,
    dist_url=None,
    args=(),
    timeout=DEFAULT_TIMEOUT
):
    world_size = nodes * gpus_per_node
    if world_size > 1:
        if dist_url == "auto":
            assert nodes == 1, f"dist_url='auto' not supported in multinode"
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
            print(f"auto mode: dist_url is {dist_url}")
        if nodes > 1 and dist_url.startswith("file://"):
            logger = logging.getLogger(__name__)
            logger.warning("file:// is not a reliable init_method for multinode. Prefer tcp://")
        
        torch.multiprocessing.spawn(
            _distributed_worker,
            nprocs=gpus_per_node,
            args=(main_func, world_size, gpus_per_node, rank, dist_url, args, timeout),
            daemon=False
        )
    else:
        main_func(*args)
    
def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    gpus_per_node,
    rank,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT
):
    assert torch.cuda.is_available(), "torch.cuda not available."
    global_rank = rank * gpus_per_node + local_rank

    try:
        torch.distributed.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Process group URL: {dist_url}")
        raise e

    comm.synchronize()

    assert gpus_per_node <= torch.cuda.device_count(), f"node {rank}: gpus_per_node {gpus_per_node} is more than the available CUDA-capable GPUs {torch.cuda.device_count()}"
    torch.cuda.set_device(local_rank)

    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // gpus_per_node

    for i in range(num_machines):
        ranks_on_i = list(range(i*gpus_per_node, (i+1)*gpus_per_node))
        pg = torch.distributed.new_group(ranks_on_i)
        if i == rank:
            comm._LOCAL_PROCESS_GROUP = pg

    main_func(args)


def prepare_train(args):
    #print(f"Namespace {args}")
    device = f"cuda:{comm.get_local_rank()}"
    net = MLP().to(device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    trainset, _ = MNIST(data_root="data")

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    # shuffle is False bc. DistributedSampler already shuffles
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, pin_memory=True, sampler=train_sampler)

    distributed_train(net, optimizer, loss_fn, args.epochs, trainloader, device, cuda_non_blocking=True)


def distributed_train(model, optimizer, loss_fn, num_epochs, dataloader, data_device, metric=accuracy, cuda_non_blocking=False):
    model.train()
    for epoch in range(num_epochs):
        loss_meter = AverageMeter()
        perf_meter = AverageMeter()
        for i, (data, labels) in enumerate(dataloader):
            print_tensor_shapes = (epoch == 0 and i == 0)
            if print_tensor_shapes:
                print(f"\tData shape in trainloader {data.shape}")

            data = data.to(data_device, non_blocking=cuda_non_blocking)
            labels = labels.to(data_device, non_blocking=cuda_non_blocking)

            batch_size = data.shape[0]

            optimizer.zero_grad()

            outputs = model(data, print_tensor_shapes).to(data_device)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            perf = metric(outputs, labels)

            loss_meter.update(loss.item(), batch_size)
            perf_meter.update(perf.item(), batch_size)
        
        total_loss = meters_avg(comm.gather(loss_meter))
        total_perf = meters_avg(comm.gather(perf_meter))
        if comm.is_main_process():
            print(f"### Epoch {epoch} || loss {total_loss} || performance {total_perf}")

def meters_avg(meters):
    meters_sum = sum([meter.sum for meter in meters])
    meters_n = sum([meter.count for meter in meters])
    return meters_sum / meters_n

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus_per_node', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-r', '--rank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument("-p", "--port", default=None, type=int, help="port to use for master process. Can be left to None when nodes=1 for auto-assignment")
    parser.add_argument('--epochs', default=3, type=int, help='number of total epochs to run')
    parser.add_argument("--batch_size", default=128, type=int, help="batch size for trainset")
    args = parser.parse_args()

    if args.nodes > 1:
        assert args.port is not None, f"If nodes > 1, then --port cannot be None"
        local_ip = socket.gethostbyname(socket.gethostname())
        dist_url = f"tcp://{local_ip}:{args.port}"
    else:
        if args.port is not None:
            dist_url = f"tcp://127.0.0.1:{args.port}"
        else:
            dist_url = "auto"
        
    print(f"dist_url is {dist_url}")

    launch(prepare_train, args.gpus_per_node, nodes=args.nodes, rank=args.rank, dist_url=dist_url, args=args)

