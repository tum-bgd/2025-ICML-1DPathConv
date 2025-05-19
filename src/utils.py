import datetime
import errno
import logging
import numpy
import os
import torch
import time
import random

from collections import defaultdict, deque
from torch.utils.flop_counter import FlopCounterMode
from typing import Union, Tuple

from model.pathconv import LayerNorm


def SetWeightDecay(model, weight_decay: float, norm_weight_decay=0.0):
    norm_classes = tuple([
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LocalResponseNorm,
        LayerNorm])
    params = {
        "other": [],
        "norm": []}
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay}

    def _add_params(module, prefix=""):
        for _, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)
    _add_params(model)
    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups


def ReduceAcrossProcesses(val):
    # no need to sync as datasets are small, dist. is unnecessary.
    return torch.tensor(val)


def CalAccuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if not e.errno == errno.EEXIST:
            raise


def SaveCheckpoint(*args, **kwargs):
    torch.save(*args, **kwargs)


def PrintAndLog(msg: str):
    print(msg)
    logging.info(msg)


def SpecifyRandomSeed(seed=13407):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def GetModelParamStat(model, report=True):
    nParam = sum(p.numel() for p in model.parameters())
    nTrainableParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if report:
        msg = f"[stat] #param: {nParam:,}\n[stat] #trainable-param: {nTrainableParam:,}"
        PrintAndLog(msg)
        return
    return nParam, nTrainableParam


def GetModelFLOPS(
    model,
    inp: Union[torch.Tensor, Tuple],
    device='cuda',
    with_backward=False,
    report=True):

    istrain = model.training
    model.eval()
    inp = inp if isinstance(inp, (torch.FloatTensor, torch.cuda.FloatTensor)) else torch.randn(inp).to(device)
    flopCounter = FlopCounterMode(mods=model, display=False, depth=None)
    with flopCounter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    totalFlops =  flopCounter.get_total_flops()
    if istrain:
        model.train()
    if report:
        msg = f"[stat] #FLOPS: {totalFlops:,} given input = {inp.size()}\n"
        PrintAndLog(msg)
        return
    return totalFlops


def MonitorGradients(model, epoch):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 10:  # Threshold for suspicious gradients
                print(f"Large gradient in {name}: {grad_norm} at step {epoch}")


class SmoothedValue:
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = ReduceAcrossProcesses([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 and i > 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    PrintAndLog(log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB,))
                else:
                    PrintAndLog(log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        msg = f"{header} Total time: {total_time_str}"
        PrintAndLog(msg)
