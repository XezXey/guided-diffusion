import numpy as np
import torch as th
import os, tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--latent_folder', type=str, required=True)
args = parser.parse_args()

if __name__ == '__main__':
    latent_path = f'{args.latent_folder}/latent.pkl'
    if not os.path.exists(latent_path):
        raise ValueError(f'Latent file not found: {latent_path}')

    latents = th.load(latent_path)

    diffae_latents = latents['conds']
    diffae_latents_mean = latents['conds_mean'] # 512
    diffae_latents_std = latents['conds_std']   # 512
    image_name = latents['image_name']
    
    print(f'Latent shape: {diffae_latents.shape}, Number of images: {len(image_name)}')
    assert diffae_latents.shape[0] == len(image_name), "Mismatch between latents and image names"
    
    # Save latents and image names separately
    # Get folder name from latent_folder path
    folder_name = os.path.basename(os.path.normpath(args.latent_folder))
    
    for i in tqdm.tqdm(range(diffae_latents.shape[0])):
        latent = diffae_latents[i]
        latent_mean = diffae_latents_mean
        latent_std = diffae_latents_std
        image_id = image_name[i].split('_')[0]
        
        out_folder = f'{args.latent_folder}/{image_id}/'
        if not os.path.exists(out_folder):
            raise ValueError(f'[#] Output folder not found: {out_folder}')
        
        if os.path.exists(f'{out_folder}/latent.npz'):
            os.remove(f'{out_folder}/latent.npz')
        if os.path.exists(f'{out_folder}/latent.pkl'):
            os.remove(f'{out_folder}/latent.pkl')
        
        # print(latent.shape, latent_mean.shape, latent_std.shape)
        # print(latent_mean, latent_std)
        out_dict = {
            'conds': latent,
            'conds_mean': latent_mean,
            'conds_std': latent_std,
            'image_name': image_name[i]
        }
        
        np.savez_compressed(
            f'{out_folder}/{image_name[i]}_latent.npz',
            conds=latent.cpu().numpy(),
            conds_mean=latent_mean.cpu().numpy(),
            conds_std=latent_std.cpu().numpy(),
            image_name=image_name[i]
        )