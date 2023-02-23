from collections import OrderedDict

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup(mode, P):
    fname = f'{P.dataset}_{P.model}_{mode}_{P.num_ways}way_{P.num_shots}shot_{P.num_shots_test}query'

    if mode == 'protonet':
        assert not P.regression

        from train.metric_based.protonet import protonet_step as train_func
        from train.metric_based.protonet import check
            
    else:
        raise NotImplementedError()

    today = check(P)
    if P.baseline:
        today = False

    # fname += f'_seed_{P.seed}'
    if P.suffix is not None:
        fname += f'_{P.suffix}'

    return train_func, fname, today


def copy_model_param(model, params=None):
    if params is None:
        params = OrderedDict(model.meta_named_parameters())
    copy_params = OrderedDict()

    for (name, param) in params.items():
        copy_params[name] = param.clone().detach()
        copy_params[name].requires_grad_()

    return copy_params


def dropout_eval(m):
    def _is_leaf(model):
        return len(list(model.children())) == 0
    if hasattr(m, 'dropout'):
        m.dropout.eval()

    for child in m.children():
        if not _is_leaf(child):
            dropout_eval(child)
