# ArcFace in PyTorch

## Data
#### Training Dataset
1. Download the [MS1MV2] dataset

Link: from [google cloud](https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view?usp=sharing)

2. Organise the dataset directory as follows:

```Shell
  ./Arcface_train_evaluate/train_tmp/faces_emore/
```

#### Evaluating Dataset
1. Download the [IJB-C] dataset

Link: from [google cloud](https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view?usp=sharing)

2. Organise the dataset directory as follows:

```Shell
  ./Arcface_train_evaluate/IJBC/
```

## Training

1. Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in ``configs/``.

2. Train the model using MS1MV2:
  ```Shell
  python train_v2.py configs/ms1mv2_mbf
  ```


## Evaluation

1. Download the [IJB-C] () dataset for evaluating

2. Run the file eval_ijbc.py

```bash
python eval_ijbc.py

