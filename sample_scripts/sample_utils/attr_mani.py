import torch as th
import numpy as np
from . import mani_utils, file_utils

class LinearClassifier(th.nn.Module):
    def __init__(self, cfg):
        super(LinearClassifier, self).__init__()
        self.cls = th.nn.Linear(cfg.param_model.light, 1)
        self.opt = th.optim.Adam(self.parameters())

    def forward(self, x):
        output = self.cls(x)
        return output
    
    def cal_loss(self, gt, pred):
        loss_fn = th.nn.BCEWithLogitsLoss()
        loss = loss_fn(pred, gt)
        return loss

    def train(self, gt, input, n_iters, progress=True):
        print(f"[#] Training Linear Classifier with iterations={n_iters}, samples_size={gt.shape[0]}")
        if progress:
            import tqdm
            t = tqdm.trange(n_iters, desc="")
        else:
            t = range(n_iters)
        for i in t:
            self.opt.zero_grad()
            pred = self.forward(input)
            loss = self.cal_loss(gt=gt, pred=pred)
            loss.backward()
            self.opt.step()
            if i % 500 == 0 and progress:
                t.set_description(f"[#] Loss = {loss}")
                t.refresh() # to show immediately the update

    def evaluate(self, gt, input):
        sigmoid = th.nn.Sigmoid()
        pred = self.forward(input.cuda().float())
        pred = sigmoid(pred) > 0.5
        accuracy = (th.sum(pred == gt.cuda()) / pred.shape[0]) * 100
        print(f"[#] Accuracy = {accuracy}")

def distance(a, b, dist_type='l2'):
    if dist_type == 'l1':
        return np.sum(np.abs(a-b))
    elif dist_type == 'l2':
        return np.sum((a-b)**2)

def retrieve_topk_params(params_set, ref_params, cfg, img_dataset_path, dist_type='l2', k=30):

    light_dist = []
    img_name_list = []
    for img_name in params_set.keys():
        light_dist.append(distance(params_set[img_name]['light'], ref_params, dist_type=dist_type))
        img_name_list.append(img_name)

    min_idx = np.argsort(light_dist)[:k]

    img_path = file_utils._list_image_files_recursively(img_dataset_path)
    img_path = [img_path[i] for i in min_idx]
    img_name = [path.split('/')[-1] for path in img_path]

    images = mani_utils.load_image(all_path=img_path, cfg=cfg, vis=True)['image']
    params_filtered = {k:params_set[k] for k in img_name}
    return images, params_filtered