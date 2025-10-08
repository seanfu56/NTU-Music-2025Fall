# NTU-Music-2025Fall
Homework of Deep Learning for Music Analysis and Generation in NTU 2025 Fall

# Environment
GPU: Nvidia GeForce RTX 3080 Ti
Driver: 570.133.07
CUDA Version

```bash=
conda create -n music-hw1 python=3.13
conda activate music-hw1
pip install -r requirements.txt
```

# Inference

0. Pretrain Weights

Manually download from 
[google drive link](https://drive.google.com/file/d/1T4oHobMQNHNdannogvpwlCBGJy_DZwdC/view?usp=drive_link)


or

```bash=
gdown 1T4oHobMQNHNdannogvpwlCBGJy_DZwdC
```

1. Preprocessing

```bash=
python eval_preprocessing.py --data_root [test_dataset_root]
```

2. Inference
```bash=
python eval_transformer.py --ckpt best_model.pt
```



# Training

1. Dataset

Ensure the format of the dataset

```
data\artist20
    test\
    train_val\
    train.json
    val.json
```

2. Preprocessing

Use demucs to remove the sounds of instruments
```bash=
python preprocess_vocals.py
```

3. Training