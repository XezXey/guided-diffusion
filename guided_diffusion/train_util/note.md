# Dataloader version (updated @ 12 Nov 2023)

## DiFaReli
### 1. cond_train_util.py
    - use for training the original DiFaReli
### 2. no_diffusion_train_util.py
    - DiFaReli without diffusion (for rebuttal, testing UNet-self reconstruction wouldn't work)

## Faster DiFaReli
### 1. ldm_train_util.py
    - use for training using latent (from LDM)
### 2. no_diffusion_train_util_paired.py
    - use for training without diffusion
    - train on paired dataset (generated data with target relit images)
### 3. cond_train_util_presample_paired.py (used by faster version of cond_train_util_paired.py)    <--- Using @ 12 Nov 2023
    - Similar to original DiFaReli
    - Can train on paired dataset (generated data with target relit images)
    - Can be selected whether denoise using source or target image
    - the faster version of cond_train_util_paired.py (actually the same script but called from different training script)
