# Update logs

Date : 26/2/2022
1. Clean repos
2. Experiment on Channels memory (Motivated from [this work](https://nvlabs.github.io/instant-ngp/))

    - Simply concat the trainable feature map (N, C, H, W) into input after GroupNorm32
    - Fixed C = 8 for all block

3. Features for adding memory (Must be defined in config.yaml)
    - add_mem: [True, True]
    The UNet-part to add the channel memory in format : [encoder, decoder] with boolean values. In this case, the channel memory were added to both encoder and decoder block
    - n_channels_mem: 8
    A constant number of channels memory to add
    More example : [ddpm_chnmem_64_decoder.yaml](./config/Uncondition_Image/ddpm_chnmem_64_decoder.yaml)

4. Network architecture that implement the channels memory

    - unet.py : 

            class UNetChnMem(): ...
            class ResBlockChnMem(): ...

Commit name : 26/2/2022 updates