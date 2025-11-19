# DeepMIR HW3

## Environment

```bash=
conda create -n music-hw3 python=3.12
conda activate music-hw3
conda install conda-forge::fluidsynth
pip install -r requirements
pip install pytorch-fast-transformers
```

## Download Dataset

```bash=
zenodo_get 1316761
unzip Pop1K7.zip
```

## Download Checkpoints

```bash=
bash download_ckpt.sh
```

## Run Inference

```bash=
bash scripts/t1_inference_1.sh
bash scripts/t1_inference_2.sh
bash scripts/t1_inference_3.sh
bash scripts/t1_inference_4.sh
bash scripts/t1_evaluation.sh

bash scripts/t2_preprocessing.sh
bash scripts/t2_inference_1.sh
bash scripts/t2_inference_2.sh
```