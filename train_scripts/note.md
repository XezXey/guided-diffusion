# Dataloader version (updated @ 12 Nov 2023)

## DiFaReli
### 1. image_train.py
    - use for training the original DiFaReli
### 2. image_train_no_dpm.py
    - DiFaReli without diffusion (for rebuttal, testing UNet-self reconstruction wouldn't work)

## Faster DiFaReli
### 1. image_train_ldm.py
    - use for training using latent (from LDM)
### 2. image_train_no_dpm_presample_paired.py
    - use for training without diffusion
    - train on paired dataset (generated data with target relit images)
### 3. image_train_presample_paired (used by faster version of image_train_paired.py)    <--- Using @ 12 Nov 2023
    - Similar to original DiFaReli
    - Can train on paired dataset (generated data with target relit images)
    - Can be selected whether denoise using source or target image
    - the faster version of cond_train_util_paired.py (actually the same script but called from different training script)
