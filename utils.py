import pickle
import random
import shutil
import sys
from datetime import datetime
import os
import time
from collections import OrderedDict, defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tensorboardX import SummaryWriter


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""

    def __init__(self, fn, ask=True, today=True, rank=0):
        self.rank = rank
        self.log_path = './logs/'

        if self.rank == 0:
            if not os.path.exists(self.log_path):
                os.mkdir(self.log_path)
            self.today = today

            logdir = self._make_dir(fn)
            if not os.path.exists(logdir):
                os.mkdir(logdir)

            if len(os.listdir(logdir)) != 0 and ask:
                ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                            "Will you proceed [y/N]? ")
                if ans in ['y', 'Y']:
                    shutil.rmtree(logdir)
                else:
                    exit(1)

            self.set_dir(logdir)

    def _make_dir(self, fn):
        if self.today:
            today = datetime.today().strftime("%y%m%d")
            logdir = self.log_path + today + '_' + fn
        else:
            logdir = self.log_path + fn
        return logdir

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.writer = SummaryWriter(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def close_writer(self):
        if self.rank == 0:
            self.writer.close()

    def log(self, string):
        if self.rank == 0:
            self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
            self.log_file.flush()

            print('[%s] %s' % (datetime.now(), string))
            sys.stdout.flush()

    def log_dirname(self, string):
        if self.rank == 0:
            self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
            self.log_file.flush()

            print('%s (%s)' % (string, self.logdir))
            sys.stdout.flush()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if self.rank == 0: self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        if self.rank == 0: self.writer.add_image(tag, images, step)

    def histo_summary(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        if self.rank == 0: self.writer.add_histogram(tag, values, step, bins='auto')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count


class AverageMeterList(object):
    """Computes and stores the average and current value of layer and bias importance"""

    def __init__(self, list_num):
        self.list_num = list_num
        self.avg_list = [AverageMeter() for _ in range(self.list_num)]

    def reset(self):
        self.avg_list = [AverageMeter() for _ in range(self.list_num)]

    def update(self, _avg_list, n=1):
        for i in range(self.list_num):
            self.avg_list[i].update(_avg_list[i], n)

    def return_average(self):
        avg_list_avg = [self.avg_list[i].average for i in range(self.list_num)]
        return avg_list_avg


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(logdir, mode='last'):
    model_path = os.path.join(logdir, f'{mode}.model')
    optim_path = os.path.join(logdir, f'{mode}.optim')
    config_path = os.path.join(logdir, f'{mode}.configs')
    lr_path = os.path.join(logdir, f'{mode}.lr')
    ema_path = os.path.join(logdir, f'{mode}.ema')

    print("=> Loading checkpoint from '{}'".format(logdir))
    if os.path.exists(model_path):
        model_state = torch.load(model_path)
        optim_state = torch.load(optim_path)
        with open(config_path, 'rb') as handle:
            cfg = pickle.load(handle)
    else:
        return None, None, None, None, None

    if os.path.exists(lr_path):
        lr_dict = torch.load(lr_path)
    else:
        lr_dict = None

    if os.path.exists(ema_path):
        ema_dict = torch.load(ema_path)
    else:
        ema_dict = None

    return model_state, optim_state, cfg, lr_dict, ema_dict


def save_checkpoint(P, step, best, model_state, optim_state, logdir, is_best=False):
    if is_best:
        prefix = 'best'
    else:
        prefix = 'last'
    last_model = os.path.join(logdir, f'{prefix}.model')
    last_optim = os.path.join(logdir, f'{prefix}.optim')
    last_config = os.path.join(logdir, f'{prefix}.configs')

    opt = {
        'step': step,
        'best': best
    }
    torch.save(model_state, last_model)
    torch.save(optim_state, last_optim)
    with open(last_config, 'wb') as handle:
        pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_checkpoint_step(P, step, best, model_state, optim_state, logdir):
    last_model = os.path.join(logdir, f'step{step}.model')
    last_optim = os.path.join(logdir, f'step{step}.optim')
    last_config = os.path.join(logdir, f'step{step}.configs')

    opt = {
        'step': step,
        'best': best
    }
    torch.save(model_state, last_model)
    torch.save(optim_state, last_optim)
    with open(last_config, 'wb') as handle:
        pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)


def cycle(loader):
    while True:
        for x in loader:
            yield x


def one_hot(ids, n_class):
    # ---------------------
    # author：ke1th
    # source：CSDN
    # artical：https://blog.csdn.net/u012436149/article/details/77017832
    """
    ids: (list, ndarray) shape:[batch_size]
    out_tensor:FloatTensor shape:[batch_size, depth]
    """

    assert len(ids.shape) == 1, 'the ids should be 1-D'

    out_tensor = torch.zeros(len(ids), n_class)

    out_tensor.scatter_(1, ids.cpu().unsqueeze(1), 1.)

    return out_tensor


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=20):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def dist_gather(P, X):
    if P.distributed:
        Xs = [torch.zeros_like(X) for _ in range(P.world_size)]
        dist.all_gather(Xs, X)
        X = torch.cat(Xs, 0)

    return X


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
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
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
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
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
