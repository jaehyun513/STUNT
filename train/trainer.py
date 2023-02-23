import time
from collections import OrderedDict

import torch
import torch.nn as nn

from common.utils import is_resume
from utils import MetricLogger, save_checkpoint, save_checkpoint_step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def meta_trainer(P, train_func, test_func, model, optimizer, train_loader, test_loader, logger):
    kwargs = {}
    kwargs_test = {}

    metric_logger = MetricLogger(delimiter="  ")

    """ resume option """
    is_best, start_step, best, acc = is_resume(P, model, optimizer)

    """ define loss function """
    criterion = nn.CrossEntropyLoss()

    """ training start """
    logger.log_dirname(f"Start training")
    for step in range(start_step, P.outer_steps + 1):

        stime = time.time()
        train_batch = next(train_loader)
        metric_logger.meters['data_time'].update(time.time() - stime)

        train_func(P, step, model, criterion, optimizer, train_batch,
                   metric_logger=metric_logger, logger=logger, **kwargs)

        """ evaluation & save the best model """
        if step % P.eval_step == 0:
            acc = test_func(P, model, test_loader, criterion, step, logger=logger, **kwargs_test)
 
            if best < acc:
                best = acc
                save_checkpoint(P, step, best, model.state_dict(),
                                optimizer.state_dict(), logger.logdir, is_best=True)

            logger.scalar_summary('eval/best_acc', best, step)
            logger.log('[EVAL] [Step %3d] [Acc %5.2f] [Best %5.2f]' % (step, acc, best))

        """ save model per save_step steps"""
        if step % P.save_step == 0:
            save_checkpoint_step(P, step, best, model.state_dict(),
                                 optimizer.state_dict(), logger.logdir)

    """ save last model"""
    save_checkpoint(P, P.outer_steps, best, model.state_dict(),
                    optimizer.state_dict(), logger.logdir)