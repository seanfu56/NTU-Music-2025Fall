# NTU-Music-2025Fall
Homework of Deep Learning for Music Analysis and Generation in NTU 2025 Fall

# Environment
GPU: Nvidia GeForce RTX 3080 Ti
Driver: 570.133.07
CUDA Version

# Inference



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