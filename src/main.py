import argparse
import datetime
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)
import os
import time
import torch
import torchvision

import utils

from model.resnet import *
from model.pathconv import *
from model.path.generator import *
from preprocess import *


def Train(model, criterion, optimizer, dLoader, device, epoch, monitor=False):
    '''
    train one epoch
    '''
    model.train()
    metricLogger = utils.MetricLogger(delimiter="  ")
    metricLogger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.7f}"))
    metricLogger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value:.2f}"))
    header = f"Epoch: [{epoch}]"
    for i, (img, tar) in enumerate(metricLogger.log_every(dLoader, len(dLoader)//3, header)):
        start_time = time.time()
        img, tar = img.to(device), tar.to(device)
        output = model(img)
        loss = criterion(output, tar)
        optimizer.zero_grad()
        loss.backward()
        if monitor:
            utils.MonitorGradients(model, i)
        # TODO: maybe add gradient clip
        optimizer.step()
        acc1, acc5 = utils.CalAccuracy(output, tar, topk=(1, 5))
        bs = img.shape[0]
        metricLogger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metricLogger.meters["acc1"].update(acc1.item(), n=bs)
        metricLogger.meters["acc5"].update(acc5.item(), n=bs)
        metricLogger.meters["img/s"].update(bs / (time.time() - start_time))


def Valid(model, criterion, dLoader, device, log_suffix=""):
    '''
    validation one epoch
    '''
    model.eval()
    metricLogger = utils.MetricLogger(delimiter="  ")
    header = f"Validation: {log_suffix}"
    nSample = 0
    with torch.inference_mode():
        for img, tar in metricLogger.log_every(dLoader, len(dLoader)//2, header):
            img = img.to(device, non_blocking=True)
            tar = tar.to(device, non_blocking=True)
            output = model(img)
            loss = criterion(output, tar)
            acc1, acc5 = utils.CalAccuracy(output, tar, topk=(1, 5))
            bs = img.shape[0]
            metricLogger.update(loss=loss.item())
            metricLogger.meters["acc1"].update(acc1.item(), n=bs)
            metricLogger.meters["acc5"].update(acc5.item(), n=bs)
            nSample += bs
    nSample = utils.ReduceAcrossProcesses(nSample)
    metricLogger.synchronize_between_processes()
    msg = f"{header} Acc@1 {metricLogger.acc1.global_avg:.3f} Acc@5 {metricLogger.acc5.global_avg:.3f}"
    utils.PrintAndLog(msg)
    return metricLogger.acc1.global_avg


device = ('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='training options')
parser.add_argument('--epoch', type=int, default=600, help='# epoch')
parser.add_argument('--bs', type=int, default=1024, help='batch size')
parser.add_argument('--dataset', type=str, default='in64', help='dataset')
parser.add_argument('--path', type=str, default='r', help='path type (r, g, og, mg, z, oz, mz)')
parser.add_argument('--model', type=str, default='s', help='model backbone')
parser.add_argument('--initlr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--opt', type=str, default='sgd', help='optimizer')
parser.add_argument('--datadir', type=str, default='/data', help='dataset directory (default: /data)')
parser.add_argument('--ckptdir', type=str, default='./ckpt', help='checkpoint directory (default: ./ckpt)')
parser.add_argument('--logdir', type=str, default='./log', help='log directory (default: ./log)')
args = parser.parse_args()

# logging
expName = f'{args.model}{args.path}' if not args.model.startswith('r') else f'{args.model}'
logDir = os.path.join(args.logdir, args.dataset)
utils.mkdir(logDir)
print(f'Logging at {logDir}')
logging.basicConfig(
    filename=os.path.join(logDir, expName),
    filemode='w',  # overwrite
    datefmt='%H:%M:%S',
    level=logging.DEBUG)

# data
if args.dataset == 'cf10':
    s = 32
    nClass = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    trTransform = GetTrTransform(s, mean, std, augment='ra', raMag=7)
    vaTransform = GetVaTransform(s, mean, std)
    trSet = torchvision.datasets.CIFAR10(root=os.path.join(args.datadir, 'cifar10'), train=True,  download=True, transform=trTransform)
    vaSet = torchvision.datasets.CIFAR10(root=os.path.join(args.datadir, 'cifar10'), train=False, download=True, transform=vaTransform)
elif args.dataset == 'svhn':
    s = 32
    nClass = 10
    mean = [0.4377, 0.4438, 0.4728]
    std = [0.1980, 0.2010, 0.1970]
    trTransform = GetTrTransform(s, mean, std, augment='ra', raMag=7)
    # trTransform = GetTrTransform(s, mean, std, augment='sv', flip=False)
    vaTransform = GetVaTransform(s, mean, std)
    trSet = torchvision.datasets.SVHN(root=os.path.join(args.datadir, 'svhn'), split='train', download=True, transform=trTransform)
    vaSet = torchvision.datasets.SVHN(root=os.path.join(args.datadir, 'svhn'), split='test',  download=True, transform=vaTransform)
elif args.dataset == 'in64':
    s = 64
    nClass = 1000
    mean = [0.482, 0.458, 0.408]
    std = [0.269, 0.261, 0.276]
    trTransform = GetTrTransform(s, mean, std, augment='ra', raMag=9)
    vaTransform = GetVaTransform(s, mean, std)
    trSet = torchvision.datasets.ImageFolder(root=os.path.join(args.datadir, 'imagenet-64', 'train'), transform=trTransform)
    vaSet = torchvision.datasets.ImageFolder(root=os.path.join(args.datadir, 'imagenet-64', 'val'  ), transform=vaTransform)
else:
    raise Exception(f'dataset {args.dataset} not supported')
trSampler = torch.utils.data.RandomSampler(trSet)
vaSampler = torch.utils.data.SequentialSampler(vaSet)
mc = GetMixupCutmix(mixup_alpha=0.2, cutmix_alpha=0.6, num_classes=nClass)
def CollateFunc(batch):
    return mc(*torch.utils.data.dataloader.default_collate(batch))
trLoader = torch.utils.data.DataLoader(
    trSet, batch_size=args.bs,
    sampler=trSampler, num_workers=4,
    pin_memory=True)
if 'in' in args.dataset:
    trLoader = torch.utils.data.DataLoader(
        trSet, batch_size=args.bs,
        sampler=trSampler, num_workers=4,
        pin_memory=True, collate_fn=CollateFunc)
vaLoader = torch.utils.data.DataLoader(
    vaSet, batch_size=args.bs,
    sampler=vaSampler, num_workers=4,
    pin_memory=True)

# path
if args.path == 'r':
    paths = GetPaths(s, s, config=[['r', 0, [0, 0], 0]])
elif args.path == 'og':
    paths = GetPaths(s, s, config=[['g', 0, [0, 0], 0]])
elif args.path == 'oz':
    paths = GetPaths(s, s, config=[['z', 0, [0, 0], 0]])
elif args.path == 'g':
    if s == 32:
        paths = GetPaths(s, s, config=[['g',  0, [ 0, 0],   0],
                                       ['g', 24, [14, 1], 180],
                                       ['g', 18, [ 7, 0], 270]])
    elif s == 64:
        paths = GetPaths(s, s, config=[['g',  0, [ 0, 0],   0],
                                       ['g',  7, [ 7, 0], 180],
                                       ['g', 40, [13, 1],   0]])
elif args.path == 'z':
    if s == 32:
        paths = GetPaths(s, s, config=[['z', 32, [16, 26], 180],
                                       ['z', 32, [14, 16],   0],
                                       ['z', 32, [12, 20],   0]])
    elif s == 64:
        paths = GetPaths(s, s, config=[['z', 64, [36, 32], 270],
                                       ['z', 64, [36, 34],  90],
                                       ['z', 64, [28, 32],   0]])
elif args.path == 'mg':
    paths = GetPaths(s, s, config=[['g', 0, [0, 0],   0],
                                   ['g', 5, [0, 4],   0],
                                   ['g', 5, [4, 0],   0],
                                   ['g', 5, [4, 4],   0],
                                   ['g', 5, [2, 2],   0],
                                   ['g', 5, [2, 4],   0]])
elif args.path == 'mz':
    if s == 32:
        paths = GetPaths(s, s, config=[['z',  0, [ 0,  0],   0],
                                       ['z', 32, [16, 26],   0],
                                       ['z', 32, [14, 16], 180],
                                       ['z', 32, [12, 20], 180],
                                       ['z', 32, [ 8,  4],   0],
                                       ['z', 32, [ 4,  8],   0]])
    elif s == 64:
        paths = GetPaths(s, s, config=[['z',  0, [ 0,  0],   0],
                                       ['z', 64, [36, 32], 270],
                                       ['z', 64, [36, 34],  90],
                                       ['z', 64, [28, 32],   0],
                                       ['z', 64, [ 4,  8],   0],
                                       ['z', 64, [ 8,  4],   0]])
    pass
else:
    raise Exception(f'path {args.path} not defined.')

# model
if args.model == 'r18':
    modelArch = ResNet18
elif args.model == 'r50':
    modelArch = ResNet50
elif args.model == 's':
    modelArch = PathConvS
elif args.model == 'b':
    modelArch = PathConvB
else:
    raise Exception('model not defined', args.model)
model = modelArch(path=paths.to(device), nClass=nClass, imgLen=s*s)
model.to(device)
utils.GetModelParamStat(model)
utils.GetModelFLOPS(model, (1, 3, s, s))

# opt
wd = 2e-04
param = utils.SetWeightDecay(model, wd, norm_weight_decay=0)
if args.opt == 'sgd':
    optimizer = torch.optim.SGD(param, lr=args.initlr, momentum=0.9, nesterov=True, weight_decay=wd)
elif args.opt == 'adamw':
    optimizer = torch.optim.AdamW(param, lr=args.initlr, weight_decay=wd)
else:
    raise Exception('optimizer not supported', args.opt)

# scheduler
nWarmupEpoch = 10
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=nWarmupEpoch),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch-nWarmupEpoch)],
    milestones=[nWarmupEpoch])

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

# checkpoints
ckptDir = os.path.join(args.ckptdir, args.dataset)
utils.mkdir(ckptDir)
print(f'Checkpoints will be saved to {ckptDir}')

# start training
best = 0.0
sTime = time.time()
for epoch in range(args.epoch):
    Train(model, criterion, optimizer, trLoader, device, epoch)
    scheduler.step()
    acc = Valid(model, criterion, vaLoader, device)
    if acc > best:
        # save better ckpt only
        ckptPath = os.path.join(ckptDir, f'ckpt_{expName}_acc1@{best:.2f}.pth')
        if not best == 0.0:
            os.remove(ckptPath)
        best = acc
        ckptPath = os.path.join(ckptDir, f'ckpt_{expName}_acc1@{best:.2f}.pth')
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "args": args}
        utils.SaveCheckpoint(checkpoint, ckptPath)
        utils.PrintAndLog(f"The best model (top-1 acc={best}) is saved to {ckptPath}")
totalTime = time.time() - sTime
formatTime = str(datetime.timedelta(seconds=int(totalTime)))
utils.PrintAndLog(f"Total training time {formatTime}")
