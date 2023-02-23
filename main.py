import sys

import torch

from torchmeta.utils.data import BatchMetaDataLoader

from common.args import parse_args
from common.utils import get_optimizer, load_model
from data.dataset import get_meta_dataset
from models.model import get_model
from train.trainer import meta_trainer
from utils import Logger, set_random_seed, cycle


def main(rank, P):
    P.rank = rank

    """ set torch device"""
    if torch.cuda.is_available():
        torch.cuda.set_device(P.rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    """ fixing randomness """
    set_random_seed(P.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    """ define dataset and dataloader """
    kwargs = {'batch_size': P.batch_size, 'shuffle': True,
              'pin_memory': True, 'num_workers': 2}
    train_set, val_set = get_meta_dataset(P, dataset=P.dataset)

    train_loader = train_set
    test_loader = val_set

    """ Initialize model, optimizer, loss_scalar (for amp) and scheduler """
    model = get_model(P, P.model).to(device)
    optimizer = get_optimizer(P, model)

    """ define train and test type """
    from train import setup as train_setup
    from evals import setup as test_setup
    train_func, fname, today = train_setup(P.mode, P)
    test_func = test_setup(P.mode, P)

    """ define logger """
    logger = Logger(fname, ask=P.resume_path is None, today=today, rank=P.rank)
    logger.log(P)
    logger.log(model)

    """ load model if necessary """
    load_model(P, model, logger)

    """ train """
    meta_trainer(P, train_func, test_func, model, optimizer, train_loader, test_loader, logger)

    """ close tensorboard """
    logger.close_writer()

 
if __name__ == "__main__":
    """ argument define """
    P = parse_args()

    P.world_size = torch.cuda.device_count()
    P.distributed = P.world_size > 1
    if P.distributed:
        print("currently, ddp is not supported, should consider transductive BN before using ddp",
              file=sys.stderr)
    else:
        main(0, P)
