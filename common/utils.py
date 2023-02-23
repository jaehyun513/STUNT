import os

import torch
import torch.optim as optim

from utils import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(P, model):
    params = model.parameters()
    optimizer = optim.Adam(params, lr=P.lr)
    return optimizer


def is_resume(P, model, optimizer):
    if P.resume_path is not None:
        model_state, optim_state, config, lr_dict, ema_dict = load_checkpoint(P.resume_path, mode='last')
        model.load_state_dict(model_state, strict=not P.no_strict)
        optimizer.load_state_dict(optim_state)
        start_step = config['step']
        best = config['best']
        is_best = False
        acc = 0.0
        if lr_dict is not None:
            P.inner_lr = lr_dict
        if ema_dict is not None:
            P.moving_average = ema_dict
    else:
        is_best = False
        start_step = 1
        best = -100.0
        acc = 0.0
    return is_best, start_step, best, acc


def load_model(P, model, logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if P.load_path is not None:
        log_(f'Load model from {P.load_path}')
        checkpoint = torch.load(P.load_path)
        if P.rank != 0:
            model.__init_low_rank__(rank=P.rank)

        model.load_state_dict(checkpoint, strict=P.no_strict)
