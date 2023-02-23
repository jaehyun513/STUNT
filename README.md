# STUNT: Self-generated Tasks from UNlabeled Tables

Official PyTorch implementation of ["STUNT: Few-shot Tabular Learning with Self-generated Tasks from Unlabeled Tables"](https://openreview.net/forum?id=_xlsjehDvlY) by Jaehyun Nam, Jihoon Tack, Kyungmin Lee, Hankook Lee, Jinwoo Shin.\
TL;DR: We propose a few-shot semi-supervised tabular learning framework that meta-learns over the self-generated tasks from unlabeled tables.

## 1. Dependencies
```
conda create -n stunt python=3.8 -y
conda activate stunt

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchmeta tensorboardX
conda install -c conda-forge faiss-gpu
```

## 2. Dataset
Download the datasets and place at `/data` folder

## 3. Training
### 3.1. Training option
The options for the training method are as follows:
- `<MODE>`: {`protonet`}
- `<MODEL>`: {`mlp`}
- `<DATASET>`: {`income`}

### 3.2. Preparing pseudo-validation by STUNT
```
cd data/<DATASET>
python generate_pseudo_val.py
```

### 3.3. Training
```
python main.py --mode <MODE> --model <MODEL> --dataset <DATASET>
```

## 4. Evaluation
Place the labeled sample index of the test set in data/<DATASET>/index<SHOT> before evaluation.
```
python eval.py --data_name <DATASET> --shot_num <SHOT> --seed <SEED> --load_path <PATH>
```

## Citation
```
```

## Reference
- [torchmeta](https://github.com/tristandeleu/pytorch-meta)
- [BOIL](https://github.com/HJ-Yoo/BOIL)
- [MetaMix](https://github.com/huaxiuyao/MetaMix)
- [Sparse-MAML](https://github.com/Johswald/learning_where_to_learn)
- [SiMT](https://github.com/jihoontack/SiMT)
