from guided_diffusion.dataloader.img_util import decolor
import matplotlib.pyplot as plt
import torch as th
from . import params_utils

def plot_sample(img, **kwargs):
    columns = 6
    rows = 17
    fig = plt.figure(figsize=(20, 20), dpi=100)
    img = img.permute(0, 2, 3, 1) # BxHxWxC
    pt = 0
    for i in range(0, img.shape[0]):
        # s_ = decolor(s=img[i], out_c='rgb')
        s_ = ((img[i] + 1) * 127.5) / 255.

        s_ = s_.detach().cpu().numpy()
        fig.add_subplot(rows, columns, pt+1)
        plt.imshow(s_)
        pt += 1

        if kwargs is not None:
            # Plot other images
            for k in kwargs:
                fig.add_subplot(rows, columns, pt+1)
                # s_ = decolor(s=kwargs[k][i].permute(1, 2, 0), out_c='rgb')
                s_ = ((kwargs[k][i].permute(1, 2, 0) + 1) * 127.5) / 255.
                s_ = s_.detach().cpu().numpy()
                plt.imshow(s_)
                pt += 1
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.65, 
                        top=0.9, 
                        wspace=0.1, 
                        hspace=0.2)
    plt.show()
    return fig

def plot_deca(sample, min_value, max_value, cfg):

    img_ = []
    from tqdm.auto import tqdm
    for i in tqdm(range(sample['deca_output'].shape[0])):
        deca_params = sample['deca_output'][i].clone()
        deca_params = params_utils.denormalize(deca_params, min_val=th.tensor(min_value).cuda(), max_val=th.tensor(max_value).cuda(), a=-cfg.param_model.bound, b=cfg.param_model.bound).float()
        shape = deca_params[None, :100]
        pose = deca_params[None, 100:106]
        exp = deca_params[None, 106:156]
        cam = deca_params[None, 156:]
        img = params_utils.params_to_model(shape=shape, exp=exp, pose=pose, cam=cam, i=i)
        img_.append(img["shape_images"])

    plot_sample(th.cat(img_, dim=0))
    return th.cat(img_, dim=0)