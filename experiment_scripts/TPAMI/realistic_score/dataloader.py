import torch as th
import pytorch_lightning as pl
import os, random, glob
from typing import Callable, Dict, List, Optional, Tuple

class RealisticScoreDataset(th.utils.data.Dataset):
    """
    Pairs real/gen frames by filename stem within each subject.
    Sampling is fixed at __init__ using the provided caps (num_subjects, frames_per_subject).
    """

    def __init__(
        self,
        real_paths: List[str],  # Paths to real data folders
        gen_paths: List[str],   # Paths to generated data folders
        num_subjects_real: int, # of subjects to sample from real data
        num_subjects_gen: int,  # of subjects to sample from generated data
        frames_per_subject: Optional[int] = None,   # Number of frames to sample per subject
        seed: Optional[int] = 47,
    ):
        pl.seed_everything(seed)
        assert len(real_paths) > 0, "No real paths provided"
        assert len(gen_paths) > 0, "No generated paths provided"

        self.real_paths = real_paths
        self.real_paths_dict = self.process_gen_paths(real_paths, num_subjects_real, frames_per_subject=1, img_ext='.jpg')
        print("[#] Total real subjects: ", sum([len(v) for v in self.real_paths_dict.values()]))
        print("[#] Total real frames: ", sum([len(v2) for v in self.real_paths_dict.values() for v2 in v.values()]))
            
        self.gen_paths = gen_paths
        self.gen_path_dict = self.process_gen_paths(gen_paths, num_subjects_gen, frames_per_subject)
        print("[#] Total generated subjects: ", sum([len(v) for v in self.gen_path_dict.values()]))
        print("[#] Total generated frames: ", sum([len(v2) for v in self.gen_path_dict.values() for v2 in v.values()]))

    def process_gen_paths(self, paths: List[str], num_subjects: int, frames_per_subject: Optional[int], img_ext: Optional[str] = '.png') -> Dict[str, List[str]]:
        """
        Process the list of paths into a dictionary mapping subject IDs to their corresponding file paths.
        Assumes that the subject ID can be extracted from the filename.
        """
        path_dict = {}
        for path in paths:
            # Available folder for each subjects
            path_dict[path] = random.sample(os.listdir(path), num_subjects)
            tmp_dict = {}
            for subject in path_dict[path]:
                subject_path = glob.glob(os.path.join(path, subject, f'*{img_ext}'))
                if frames_per_subject is not None:
                    subject_path = random.sample(subject_path, min(frames_per_subject, len(subject_path)))
                tmp_dict[subject] = subject_path
            path_dict[path] = tmp_dict
                
        return path_dict


if __name__ == "__main__":
    # Example usage
    real_data_paths = ['/data/mint/DPM_Dataset/TPAMI_MajorRevision/Realsitric_Score/real_images/ffhq_256/train/']
    gen_data_paths = ['/data/mint/DPM_Dataset/TPAMI_MajorRevision/Realsitric_Score/gen_images/log=paired+difareli+cs+nodpm+trainset_256_cfg=paired+difareli+cs+nodpm+trainset_256.yaml_rot1_srcC/', 
                      '/data/mint/DPM_Dataset/TPAMI_MajorRevision/Realsitric_Score/gen_images/log=paired+difareli+cs+nodpm+trainset_256_cfg=paired+difareli+cs+nodpm+trainset_256.yaml_rot1_maxC/'
                    ]
    dataset = RealisticScoreDataset(real_data_paths, gen_data_paths, num_subjects_real=10, num_subjects_gen=10, frames_per_subject=5)