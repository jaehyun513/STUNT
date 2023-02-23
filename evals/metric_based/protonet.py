import torch

from torchmeta.utils.prototype import get_prototypes

from train.metric_based import get_accuracy
from utils import MetricLogger
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check(P):
    filename_with_today_date = True
    return filename_with_today_date


def test_classifier(P, model, loader, criterion, steps, logger=None):
    metric_logger = MetricLogger(delimiter="  ")

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for n, batch in enumerate(loader):
        if n * P.test_batch_size > P.max_test_task:
            break

        train_inputs, train_targets = batch['train']

        num_ways = len(set(list(train_targets[0].numpy())))
        train_inputs = train_inputs.to(device)
        train_targets = train_targets.to(device)
        with torch.no_grad():
            train_embeddings = model(train_inputs)

        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.to(device)
        test_targets = test_targets.to(device)
        with torch.no_grad():
            test_embeddings = model(test_inputs)

        prototypes = get_prototypes(train_embeddings, train_targets, num_ways)

        squared_distances = torch.sum((prototypes.unsqueeze(2)
                                    - test_embeddings.unsqueeze(1)) ** 2, dim=-1)
        loss = criterion(-squared_distances, test_targets)

        acc = get_accuracy(prototypes, test_embeddings, test_targets).item()

        metric_logger.meters['loss'].update(loss.item())
        metric_logger.meters['acc'].update(acc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    log_(' * [Acc@1 %.3f] [LossOut %.3f]' %
         (metric_logger.acc.global_avg, metric_logger.loss.global_avg))

    if logger is not None:
        logger.scalar_summary('eval/acc', metric_logger.acc.global_avg, steps)
        logger.scalar_summary('eval/loss_test', metric_logger.loss.global_avg, steps)

    model.train(mode)

    return metric_logger.acc.global_avg
