# pytorch_distributed_playground

PyTorch scripts to get acquainted with the GPU parallelization modalities in PyTorch.

### REQUIREMENTS

Requires PyTorch >= 1.8.

Moreover, the  script `playground/distributed_data_parallel2.py` requires the package `detectron2`.

## Data parallelism

Data parallelism refers to the parallelization of the computation (for training/inference) by scattering the batch of data on multiple GPUs.

### DataParallel

`DataParallel` is a simple paradigm which can be used on a **single machine** with multiple GPUs.

The only required step w.r.t. a regular PyTorch GPU training is to wrap a `torch.nn.Module` (already sent to a GPU) into a `torch.nn.DataParallel` module.
The original GPU will act as a "master process" for the accumulation of the gradient.

### DistributedDataParallel

`DistributedDataParallel` enables us to run a PyTorch script on multiple GPUs **AND** machines. Despite being more efficient than `DataParallel` itself, it requires some additional steps for setting it up.

Namely, the machines and processes need to be able to communicate between themselves, hence we need to establish an IP and a port used for the communications.

Each machine spawns n processes (one per GPU), then each of these processes launches a pre-defined function (e.g. for training, inference...).

The spawning is performed by `torch.multiprocessing.spawn(<function>, nprocs=<GPUs per node>, args=<arguments for function>)`, where `function` is the function (e.g., training) that each process will run and `args` its arguments.

Inside `function`, we call `torch.distributed.init_process_group`, which accepts as parameters:
* the backend to be used for the parallelization (devs suggest to use `nccl`)
* the `init_method`, i.e., the location where to find the info on the IP & port of the master process
* the world size (the total number of machines)
* the rank (index) of the current node

This method is tasked with setting up the communications between the various nodes.

Then, we need to wrap the model around a `DistributedDataParallel` structure and to construct the `DataLoaders` by specifying a `DistributedSampler` which allows the processes to be aware of the presence of each other (even on different machines) when creating the mini-batches of data (i.e., we avoid that different processes *see* the same batch of data during the same epoch).

## Model parallelism

Model parallelism refers to the splitting of the model in multiple, sequential "chunks" which are each processed in different GPUs.
It's usually employed when a model is too large to fit in a single GPU.

As opposed to data parallelism paradigms, PyTorch offers no plug-and-play API; rather, the model splitting and subsequent implementation of `forward` have to be done "by hand":

1. During the initialization of the module, we need to define the chunks and send them to their respective GPU. For instance:

   `self.chunk2 = nn.Sequential(<modules>).to(self.device[2])`
2. During the forward method, we need to send the output of the previous chunk to the GPU in which the next chunk resides. For instance, with reference to point 1:

    `out = self.chunk1(x).to(self.device[2])`

In order to minimize idle times of GPUs we can make use of batch pipelining, i.e., the mini-batch is split into multiple smaller batches:
* batch `m` is being passed through `chunk2` in GPU1
* at the same time, batch `m+1` is being fed to `chunk1` in GPU0

## Scripts

The mains scripts for this implementation are found in the `playground/` folder.
In all these programs, we train a simple MultiLayer Perceptron (MLP) on MNIST for 3 epochs.

* `single_gpu.py` shows a regular implementation with training on a single GPU and serves as a "foundation" for the more complex scripts;
* `data_parallel.py` shows how the example on the single gpu can be expanded on multiple GPUs on the same machine by using the `DataParallel` structure;
* `distributed_data_parallel.py` extends the example for `DistributedDataParallel`. For instance, to run on 2 nodes with 2 GPUs each, specify, on the first node:

    `python playground/distributed_data_parallel.py -n 2 -g 2 -nr 0 -ip <IP of first node>`

    Then, on the second node:

    `python playground/distributed_data_parallel.py -n 2 -g 2 -nr 1 -ip <IP of first node>`

* `distributed_data_parallel2.py` showcases how `DistributedDataParallel` can be enhanced by the FAIR package `detectron2` for handling the communications between the processes (also for the training metrics calculation). Run like before, but:
  * `-nr` becomes `-r`
  * there's an additional argument `--port`, the port number for the communication between processes (in the previous script it was `8888` by default)
  * there's also the possibility to save a checkpoint of the model for each epoch, by specifying a path for this checkpoint with the `--savefile` (`-s`) argument.
* `model_parallel.py` shows an implementation of the model parallel paradigm, both with and without pipelining. Run with the argument `--pipelined` if you want the use the batch pipelining.

### Practical notes for running `DistributedDataParallel` on multiple nodes of a cluster

If you're unable to launch two concurrent interactive jobs on different nodes, one possible workaround (with PBS scheduler) is the following:

* launch an interactive job on two nodes:
  
  `qsub -I -l nodes=node1+node2,walltime=...`
* `ssh` into `node1` (`ssh username@node1`), retrieve its IP, and launch the script for the first machine
  
  `python playground/distributed_data_parallel.py -n 2 -g 2 -nr 0 -ip <IP of first node>`
* `ssh` into `node2` and launch the script for the second machine (no need to retrieve its IP now)

  `python playground/distributed_data_parallel.py -n 2 -g 2 -nr 1 -ip <IP of first node>`

Although, I'm sure there exist more elegant and efficient work-arounds 😛

## References

This repo is adapted from various docs and blog posts:

**DataParallel**
* https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
* https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
  
**DistributedDataParallel**
* https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
* https://towardsdatascience.com/how-to-convert-a-pytorch-dataparallel-project-to-use-distributeddataparallel-b84632eed0f6
  
**Model parallelization**
* https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html
