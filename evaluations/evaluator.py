import torch as th
import numpy as np
import argparse
import matplotlib.pyplot as plt
from eval_dataloader import eval_loader

class Evaluator():
    def __init__(self, device='cuda'):
        # LPIPS
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
        # MSE
        # from torchmetrics import MeanSquaredError
        # self.mse = MeanSquaredError()
        
        # SSIM & DSSIM
        from torchmetrics import StructuralSimilarityIndexMeasure
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, return_full_image=True).to(device)
        
    def compute_mse(self, gt, pred, mask):
        err = ((pred-gt)*mask)**2
        out = th.sum(err) / th.sum(mask)
        return out

    def compute_ssim_dssim(self, gt, pred, mask):
        _, ssim_map = self.ssim(pred, gt)
        ssim_score = th.sum(ssim_map * mask) / th.sum(mask);
        dssim_score = (1 - ssim_score) / 2 
        return ssim_score, dssim_score
        
    def evaluate(self, pred, gt):
        #LPIPS 
        lpips_score = self.lpips(img1=pred, img2=gt)
        
        #SSIM
        ssim_score, ssim_map = self.ssim(pred, gt)
        
        #MSE
        mse_score = self.mse(pred, gt)
        
        return {'lpips':lpips_score,
                'ssim':ssim_score, 
                'mse':mse_score
        }
        
    def evaluate_each(self, pred, gt, mask=None, score_dict=None):
        assert pred.shape[0] == gt.shape[0]
        B = gt.shape[0]
        
        # Init if first sub_batch
        if score_dict is None:
            score_dict = {
                'lpips':[],
                'ssim':[], 
                'dssim':[], 
                'mse':[]
            }
            
        for i in range(B):  # Per image
            pred_ = pred[[i]]
            gt_ = gt[[i]] 
            mask_ = mask[[i]] 
            #TODO: LPIPS with mask
            lpips_score = self.lpips(img1=pred_, img2=gt_)
            
            # SSIM & DSSIM
            ssim_score, dssim_score = self.compute_ssim_dssim(pred=pred_, gt=gt_, mask=mask_)
            
            # MSE
            mse_score = self.compute_mse(pred=pred_, gt=gt_, mask=mask_)
            
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
            print(f'\t {k} : {th.mean(v)} +- {th.std(v)}')
        
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", help="path to groundtruth folder")
    parser.add_argument("--pred", help="path to prediction folder")
    parser.add_argument("--mask", help="path to mask folder")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--upsampling", action='store_true', default=False)
    
    args = parser.parse_args()
    
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
    for i, sub_batch in enumerate(loader):
        print(f"[#] Evaluating : {sub_batch['img_name']}")
        sub_gt = sub_batch['gt'].float()
        sub_pred = sub_batch['pred'].float()
        sub_mask = sub_batch['mask'].float()
        if args.upsampling:
            sub_pred = upsample(sub_pred)
        # plot = (th.cat((sub_gt[0].permute(1, 2, 0), sub_pred[0].permute(1, 2, 0), sub_mask[0].permute(1, 2, 0)), dim=1).cpu().numpy() * 255).astype(np.uint8)
        # plt.imshow(plot)
        # plt.title('GT Vs. Prediction')
        # plt.savefig('./gg.png')
        score_dict = eval.evaluate_each(pred=sub_pred.cuda(), gt=sub_gt.cuda(), mask=sub_mask.cuda(), score_dict=score_dict)
        sub_pred = sub_pred.detach()
        sub_gt = sub_gt.detach()
        sub_mask = sub_mask.detach()
    eval.print_score(score_dict)
    

 
   
if __name__ == "__main__":
    main()