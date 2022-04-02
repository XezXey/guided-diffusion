import glob
from config.base_config import parse_args
import torch as th
from guided_diffusion.script_util import (
    create_img_and_diffusion,
)

class CkptLoader():
    def __init__(self, log_dir, cfg_name) -> None:
        self.sshfs_mount_path = "/data/mint/model_logs_mount/"
        self.sshfs_path = "/data/mint/model_logs/"

        self.log_dir = log_dir
        self.cfg_name = cfg_name
        self.model_path = self.get_model_path()
        self.cfg = self.get_cfg()
        self.name = self.cfg.img_model.name

    # Config file
    def get_cfg(self,):
        cfg_file_path = glob.glob("../config/*/*", recursive=True)
        cfg_file_path = [cfg_path for cfg_path in cfg_file_path if f"/{self.cfg_name}" in cfg_path]    # Add /{}/ to achieve a case-sensitive of folder
        assert len(cfg_file_path) <= 1
        cfg_file = cfg_file_path[0]
        cfg = parse_args(ipynb={'mode':True, 'cfg':cfg_file})
        return cfg

    # Log & Checkpoint file 
    def get_model_path(self,):
        model_logs_path = glob.glob(f"{self.sshfs_mount_path}/*/*/", recursive=True) + glob.glob(f"{self.sshfs_path}/*/*/", recursive=True)
        model_path = [m_log for m_log in model_logs_path if f"/{self.log_dir}/" in m_log]    # Add /{}/ to achieve a case-sensitive of folder
        print(model_path)
        assert len(model_path) <= 1
        return model_path[0]

    # Load model
    def load_model(self, ckpt_selector, step):
        if ckpt_selector == "ema":
            ckpt = f"ema_0.9999_{step}"
        elif ckpt_selector == "model":
            ckpt = f"model{step}"
        else: raise NotImplementedError

        self.available_model()


        img_model_path = f"{self.model_path}/{self.name}_{ckpt}.pt"
        model_dict, diffusion = create_img_and_diffusion(self.cfg)
        model_dict = {k: v for k, v in img_model.items() if v is not None}


        img_model = model_dict[self.cfg.img_model.name]
        img_model.load_state_dict(
            th.load(img_model_path, map_location="cpu")
        )
        img_model.to('cuda')
        img_model.eval()

        return img_model, diffusion

    def available_model(self):

        import re
        avail_m = glob.glob(f"{self.model_path}/*.pt")
        filtered_m = []
        for m in avail_m:
            r = re.search(r"(_(\d+).pt)", m)
            if r:
                filtered_m.append(list(r.groups())[0])
        print("Available ckpt : ", sorted(filtered_m))
