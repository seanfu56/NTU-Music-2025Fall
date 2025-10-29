# NTU-Music-2025Fall
Homework of Deep Learning for Music Analysis and Generation in NTU 2025 Fall

# Environment
GPU: Nvidia GeForce RTX 4070 Ti
Driver: 525.147.05
CUDA Version

```bash=
conda create -n music-hw2 python=3.11
conda activate music-hw2
pip install -r requirements.txt
```

# Download Data

```bash=
bash download.sh
```

# Run Task 1: Retrieval
```bash=
python retrieval.py
python retrieval_benchmark.py
```

Results would be saved in output/retrieval

# Run Task 2: Generation

```bash=
python generation_caption.py
python generation.py
python generation_melody.py
python generation_benchmark.py
```

Results would be saved in output/generation