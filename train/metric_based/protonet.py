import time

import torch
import torch.nn.functional as F
import numpy as np

from torchmeta.utils.prototype import get_prototypes

from train.metric_based import get_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check(P):
    filename_with_today_date = True
    assert P.num_shots_global == 0
    return filename_with_today_date


def protonet_step(P, step, model, criterion, optimizer, batch, metric_logger, logger):
    
    stime = time.time()
    model.train()

    assert not P.regression

    train_inputs, train_targets = batch['train']
    num_ways = len(set(list(train_targets[0].numpy())))
    test_inputs, test_targets = batch['test']
        

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    train_embeddings = model(train_inputs)

    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)
    test_embeddings = model(test_inputs)

    prototypes = get_prototypes(train_embeddings, train_targets, num_ways)
    
    squared_distances = torch.sum((prototypes.unsqueeze(2)
                                - test_embeddings.unsqueeze(1)) ** 2, dim=-1)
    loss = criterion(-squared_distances, test_targets)

    """ outer gradient step """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = get_accuracy(prototypes, test_embeddings, test_targets).item()


    """ track stat """
    metric_logger.meters['batch_time'].update(time.time() - stime)
    metric_logger.meters['meta_test_cls'].update(loss.item())
    metric_logger.meters['train_acc'].update(acc)

    if step % P.print_step == 0:
        logger.log_dirname(f"Step {step}")
        logger.scalar_summary('train/meta_test_cls',
                              metric_logger.meta_test_cls.global_avg, step)
        logger.scalar_summary('train/train_acc',
                              metric_logger.train_acc.global_avg, step)
        logger.scalar_summary('train/batch_time',
                              metric_logger.batch_time.global_avg, step)

        logger.log('[TRAIN] [Step %3d] [Time %.3f] [Data %.3f] '
                   '[MetaTestLoss %f]' %
                   (step, metric_logger.batch_time.global_avg, metric_logger.data_time.global_avg,
                    metric_logger.meta_test_cls.global_avg))
