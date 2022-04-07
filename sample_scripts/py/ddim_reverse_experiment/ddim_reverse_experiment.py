import subprocess
import itertools
import argparse
import math
import multiprocessing as mp

parser = argparse.ArgumentParser()
# parser.add_argument('--available_gpus', type=int, required=True)
args = parser.parse_args()

def run_each_settings(cmd, gpu_id):
    cmd += f" --gpu {gpu_id}"
    subprocess.run(cmd.split(' '))

def driver(cmd_chunk, gpu_id):
    p = mp.Pool(1)
    p.starmap(run_each_settings, zip(cmd_chunk, [gpu_id]*len(cmd_chunk)))

if __name__ == '__main__':
    log_cfg_pair = {'cond_img64_by_deca_arcface.yaml':'cond_img64_by_deca_arcface'}
    sets = ['valid']
    steps = [300000, 500000]
    out_dir = "./sampling_results/ddim_reverse/"
    ckpts = ['ema']
    bidx = list(range(21))
    interp_idx = [[12, 10], [9, 15], [26, 13], [23, 9]]

    interpolate = [['pose']]
    all_comb_itp = []
    for i in range(1, len(interpolate) + 1):
        all_comb_itp.append(list(itertools.combinations(interpolate, i)))
    all_comb_itp = list(itertools.chain(*all_comb_itp))
    cmd_all = []
    for cfg, log in log_cfg_pair.items():
        for itp in all_comb_itp:
            itp = list(itertools.chain(*itp))
            itp = ' '.join(itp)
            for step in steps:
                for ckpt in ckpts:
                    for set in sets:
                        for b in bidx:
                            for itp_idx in interp_idx:
                                cmd = f"python sampling_ddim_reverse.py --set={set} --step={step} --log_dir={log} --cfg_name={cfg} --ckpt_selector={ckpt} --interpolate {itp} --out_dir={out_dir} --base_idx={b} --src_idx={itp_idx[0]} --dst_idx={itp_idx[1]}"
                                cmd_all.append(cmd)
    mp.Process(target=driver, args=(cmd_all, '1,2,3')).start()
    # all_p = []
    # for i in range(n_chunk):
    #     all_p.append(mp.Process(target=driver, args=(cmd_chunk[i], i)))
    #     all_p[i].start()
    

