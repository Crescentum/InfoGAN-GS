# InfoGAN — PyTorch Reproduction

## Environment Setup

### 1. Create conda environment

```bash
conda create -n ee142 python=3.10 -y
conda activate ee142
```

### 2. Install PyTorch (CUDA 11.8)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install dependencies

```bash
pip install numpy matplotlib tqdm tensorboard scikit-learn scipy
```

**requirements.txt:**

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
tensorboard>=2.13.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

> **Note:** If other versions of dependencies can work, it is also ok to use, this requirement is just an example.
---

## Dataset Preparation

### Step 1 — Download (on a machine with internet)

```bash
python download.py
```

This downloads MNIST, SVHN, and CelebA into `./data/`.

### Step 2 — Verify

```bash
ls ~/EE142/data/MNIST/raw/          # should show 8 files
ls ~/EE142/data/                    # should show train_32x32.mat, test_32x32.mat
ls ~/EE142/data/celeba/             # should show img_align_celeba/ and .txt files
```

Expected layout:

```
data/
├── MNIST/
│   └── raw/
│       ├── train-images-idx3-ubyte
│       ├── train-labels-idx1-ubyte
│       ├── t10k-images-idx3-ubyte
│       ├── t10k-labels-idx1-ubyte
│       └── *.gz
├── train_32x32.mat
├── test_32x32.mat
└── celeba/
    ├── img_align_celeba/
    ├── list_attr_celeba.txt
    └── list_eval_partition.txt
```

## CelebA Experiments

CelebA currently uses the dedicated scripts below:

- model: `model_celeba.py`
- trainer: `trainer_celeba.py`
- training entry point: `train_celeba.py`
- visualization: `visualize_celeba.py`
- TensorBoard export: `export_results_celeba.py`

Current CelebA runs support only vanilla InfoGAN. InfoNCE and WGAN-GP modes are not part of the CelebA setup.

### Train from scratch

```bash
python train_celeba.py --mode vanilla --epochs 10 --updates_per_epoch 200 --batch_size 128 --lambda_disc 1.0 --lambda_cont 0 --n_critic 1 --lr_g 0.0002 --lr_d 0.0002 --ckpt_dir checkpoints/celeba_stage4_single_code_infogan --log_dir logs/celeba_stage4_single_code_infogan
```

### Resume training

```bash
python train_celeba.py --mode vanilla --epochs 50 --updates_per_epoch 200 --batch_size 128 --resume checkpoints/celeba_stage4_single_code_infogan/celeba_vanilla_final.pt --ckpt_dir checkpoints/celeba_stage4_single_code_infogan --log_dir logs/celeba_stage4_single_code_infogan
```

### Generate latent traversal images

```bash
python visualize_celeba.py --ckpt checkpoints/celeba_stage4_single_code_infogan/celeba_vanilla_final.pt --out_dir results/celeba_stage4_single_code_infogan_from_ckpt
```

### Export loss curves and TensorBoard images

```bash
python export_results_celeba.py --log_dir logs/celeba_stage4_single_code_infogan --out_dir results/celeba_stage4_single_code_infogan_exported
```
