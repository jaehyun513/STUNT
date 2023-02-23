import torch
from torchvision import transforms

from torchmeta.transforms import ClassSplitter, Categorical

from data.income import Income

def get_meta_dataset(P, dataset, only_test=False):

    if dataset == 'income':
        meta_train_dataset = Income(tabular_size = 105,
                                    seed=P.seed,
                                    source='train',
                                    shot=P.num_shots,
                                    tasks_per_batch=P.batch_size,
                                    test_num_way = P.num_ways,
                                    query = P.num_shots_test)

        meta_val_dataset = Income(tabular_size = 105,
                                    seed=P.seed,
                                    source='val',
                                    shot=1, 
                                    tasks_per_batch=P.test_batch_size,
                                    test_num_way = 2,
                                    query = 30)

    else:
        raise NotImplementedError()

    return meta_train_dataset, meta_val_dataset
