from matplotlib.bezier import get_parallels
import torch as th
import numpy as np
import argparse
from eval_dataloader import eval_loader

class Evaluator():
    def __init__(self):
        # LPIPS
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        # MSE
        from torchmetrics import MeanSquaredError
        self.mse = MeanSquaredError()
        
        # SSIM & DSSIM
        from torchmetrics import StructuralSimilarityIndexMeasure
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, return_full_image=True)
        
    def evaluate(self, pred, gt):
        lpips_score = self.lpips(img1=pred, img2=gt)
        ssim_score, ssim_map = self.ssim(pred, gt)
        mse_score = self.mse(pred, gt)
        return {'lpips':lpips_score,
                'ssim':ssim_score, 
                'mse':mse_score
        }
        
    def evaluate_each(self, pred, gt, mask=None):
        assert pred.shape[0] == gt.shape[0]
        B = gt.shape[0]
        score_dict = {
            'lpips':[],
            'ssim':[], 
            'dssim':[], 
            'mse':[]
        }
        if mask is not None:
            pred_ = pred_ * mask
            gt_ = gt_ * mask
            
        for i in range(B):
            pred_ = pred[[i]]
            gt_ = gt[[i]] 
            # Score & Map
            lpips_score = self.lpips(img1=pred_, img2=gt_)
            ssim_score, ssim_map = self.ssim(pred_, gt_)
            dssim_score = (1 - ssim_score) / 2 
            mse_score = self.mse(pred_, gt_)
            
            # Save to dict
            score_dict['lpips'].append(lpips_score)
            score_dict['ssim'].append(ssim_score)
            score_dict['dssim'].append(dssim_score)
            score_dict['mse'].append(mse_score)
            
        return score_dict
        
    def print_score(self, score_dict):
        print("[#] Evaluation Score")
        for k, v in score_dict.items():
            v = th.tensor(v)
            print(f'{k} : {th.mean(v)} +- {th.std(v)}')
        
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", help="path to groundtruth folder")
    parser.add_argument("--pred", help="path to prediction folder")
    parser.add_argument("--batch_size", type=int, help="batch size")
    args = parser.parse_args()
    
    print("[#] DPM - Relit Evaluation")
    print(f"[#] Ground truth : {args.gt}")
    print(f"[#] Prediction : {args.pred}")
    
    loader, dataset = eval_loader(gt_path=args.gt, 
                                  pred_path=args.pred, 
                                  batch_size=args.batch_size
                                )
    eval = Evaluator()
    
    for sub_batch in iter(loader):
        sub_gt = sub_batch['gt'].float()
        sub_pred = sub_batch['pred'].float()
        score = eval.evaluate_each(pred=sub_pred, gt=sub_gt)
        eval.print_score(score)
    

 
   
if __name__ == "__main__":
    main()