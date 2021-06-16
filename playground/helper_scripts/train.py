import torch

class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(outputs, labels):
    output_classes = outputs.argmax(dim=1)
    correct = output_classes.eq_(labels).sum()
    return correct / labels.shape[0]


def train(model, optimizer, loss_fn, num_epochs, dataloader, device, metric=accuracy):
    model.train()
    for epoch in range(num_epochs):
        loss_meter = AverageMeter()
        perf_meter = AverageMeter()
        for i, (data, labels) in enumerate(dataloader):
            print(f"\tData shape in trainloader {data.shape}")
            data = data.to(device)
            labels = labels.to(device)

            batch_size = data.shape[0]

            optimizer.zero_grad()

            outputs = model(data)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            perf = metric(outputs, labels)

            loss_meter.update(loss.item(), batch_size)
            perf_meter.update(perf.item(), batch_size)
        
        print(f"### Epoch {epoch} || loss {loss_meter.avg} || performance {perf_meter.avg}")
            