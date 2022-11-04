import torch as th
import numpy as np
import argparse
import lpips
import matplotlib.pyplot as plt
from eval_dataloader import eval_loader
import os
from pathlib import Path
import json


parser = argparse.ArgumentParser()
parser.add_argument("--gt", help="path to groundtruth folder")
parser.add_argument("--pred", help="path to prediction folder")
parser.add_argument("--mask", help="path to mask folder")
parser.add_argument("--batch_size", type=int, help="batch size")
parser.add_argument("--upsampling", action='store_true', default=False)

args = parser.parse_args()

class Evaluator():
    def __init__(self, device='cuda'):
        # LPIPS
        # from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        # self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
        self.lpips = lpips.LPIPS(net='alex', spatial=True).to(device)
        
        # SSIM & DSSIM
        from torchmetrics import StructuralSimilarityIndexMeasure
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, return_full_image=True).to(device)
        
        # Score dict
        self.score_dict = {
            'lpips':[],
            'ssim':[], 
            'dssim':[], 
            'mse':[],
            'n_samples': 0
        }
        
        self.img_score_dict = {'each_image':{}}
        
    def compute_mse(self, gt, pred, mask):
        err = ((pred-gt)*mask)**2
        out = th.sum(err) / th.sum(mask)
        return out

    def compute_ssim_dssim(self, gt, pred, mask):
        _, ssim_map = self.ssim(pred, gt)
        ssim_score = th.sum(ssim_map * mask) / th.sum(mask);
        dssim_score = (1 - ssim_score) / 2 
        return ssim_score, dssim_score

    def compute_lpips(self, gt, pred, mask):
        gt = (gt.clone() * 2) - 1
        pred = (pred.clone() * 2) - 1
        # print(th.max(pred))
        # print(th.min(pred))

        # print(th.max(gt))
        # print(th.min(gt))
        # exit()
        lpips_score = self.lpips.forward(gt, pred)
        lpips_score = th.sum(mask * lpips_score) / th.sum(mask)
        return lpips_score
        
    def evaluate_each(self, pred, gt, name, mask=None):
        assert pred.shape[0] == gt.shape[0]
        B = gt.shape[0]
        
        with th.no_grad():
            for i in range(B):  # Per image
                pred_ = pred[[i]]
                gt_ = gt[[i]]
                mask_ = mask[[i]]
                assert mask_.shape[1] == 3
                name_ = name[i]
                
                # LPIPS
                lpips_score = self.compute_lpips(pred=pred_, gt=gt_, mask=mask_)
                
                # SSIM & DSSIM
                ssim_score, dssim_score = self.compute_ssim_dssim(pred=pred_, gt=gt_, mask=mask_)
                
                # MSE
                mse_score = self.compute_mse(pred=pred_, gt=gt_, mask=mask_)
                
                # Save to dict
                self.score_dict['lpips'].append(lpips_score)
                self.score_dict['ssim'].append(ssim_score)
                self.score_dict['dssim'].append(dssim_score)
                self.score_dict['mse'].append(mse_score)
                self.score_dict['n_samples'] += 1
                
                # Each image score
                self.img_score_dict['each_image'][name_] = {
                    'lpips':str(lpips_score.cpu().numpy()),
                    'ssim':str(ssim_score.cpu().numpy()), 
                    'dssim':str(dssim_score.cpu().numpy()),
                    'mse':str(mse_score.cpu().numpy()),
                }
            
    def print_score(self):
        print("[#] Evaluation Score")
        print(f"[#] Total samples = {self.score_dict['n_samples']}")
        for k, v in self.score_dict.items():
            if k == 'n_samples': continue
            v = th.tensor(v)
            print(f'\t {k} : {th.mean(v)} +- {th.std(v)}')
            
    def save_score(self):
        self.img_score_dict['running_command'] = f"--gt {args.gt} --pred {args.pred} --mask {args.mask} --batch_size {args.batch_size}, --upsampling {args.upsampling}"
        self.img_score_dict['eval_score'] = {}
        print("[#] Saving Evaluation Score...")
        for k, v in self.score_dict.items():
            if k == 'n_samples':
              self.img_score_dict['eval_score'][k] = f'{v}'
            else:
              v = th.tensor(v)
              self.img_score_dict['eval_score'][k] = f'{th.mean(v)} +- {th.std(v)}'
        save_path = Path(args.pred).parents[0]
        with open(f'{save_path}/eval_score.json', 'w') as jf:
            json.dump(self.img_score_dict, jf, indent=4)
            
def main():
    
    print("[#] DPM - Relit Evaluation")
    print(f"[#] Ground truth : {args.gt}")
    print(f"[#] Prediction : {args.pred}")
    
    loader, dataset = eval_loader(gt_path=args.gt, 
                                  pred_path=args.pred, 
                                  mask_path=args.mask,
                                  batch_size=args.batch_size
                                )
    eval = Evaluator()
    upsample = th.nn.UpsamplingBilinear2d(scale_factor=2)
    
    score_dict = None
    for sub_batch in loader:
        print(f"[#] Evaluating : {sub_batch['img_name']}")
        sub_gt = sub_batch['gt'].float()
        sub_pred = sub_batch['pred'].float()
        sub_mask = sub_batch['mask'].float()
        sub_name = sub_batch['img_name']
        if args.upsampling:
            sub_pred = upsample(sub_pred)
        # plot = (th.cat((sub_gt[0].permute(1, 2, 0), sub_pred[0].permute(1, 2, 0), sub_mask[0].permute(1, 2, 0)), dim=1).cpu().numpy() * 255).astype(np.uint8)
        # plt.imshow(plot)
        # plt.title('GT Vs. Prediction')
        # plt.savefig('./tmp_img/gg.png')
        eval.evaluate_each(pred=sub_pred.cuda(), gt=sub_gt.cuda(), mask=sub_mask.cuda(), name=sub_name)
        sub_pred = sub_pred.detach()
        sub_gt = sub_gt.detach()
        sub_mask = sub_mask.detach()
    
    eval.print_score()
    eval.save_score()
    

 
   
if __name__ == "__main__":
    main()
