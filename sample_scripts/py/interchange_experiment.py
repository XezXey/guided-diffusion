import subprocess
import itertools
import argparse
import math
import multiprocessing as mp
parser = argparse.ArgumentParser()
parser.add_argument('--available_gpus', type=int, required=True)
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
    out_dir = "./sampling_results/interchange/"
    ckpts = ['ema']

    interchange = [['pose'], ['exp'], ['light'], ['shape'], ['faceemb'], ['cam']]
    all_comb_itc = []
    for i in range(1, len(interchange) + 1):
        all_comb_itc.append(list(itertools.combinations(interchange, i)))
    all_comb_itc = list(itertools.chain(*all_comb_itc))

    cmd_all = []
    for cfg, log in log_cfg_pair.items():
        for itc in all_comb_itc:
            itc = list(itertools.chain(*itc))
            itc = ' '.join(itc)
            for step in steps:
                for ckpt in ckpts:
                    for set in sets:
                        cmd = f"python sampling_interchange.py --set={set} --step={step} --log_dir={log} --cfg_name={cfg} --ckpt_selector={ckpt} --interchange {itc} --out_dir={out_dir}"
                        cmd_all.append(cmd)

    n_chunk = args.available_gpus
    chunk_size = math.ceil(len(cmd_all)/n_chunk)
    cmd_chunk = [cmd_all[i:i+chunk_size] for i in range(0, len(cmd_all), chunk_size)]
    assert len(cmd_chunk) == n_chunk
    assert len(cmd_all) == sum([len(x) for x in cmd_chunk])

    all_p = []
    for i in range(n_chunk):
        all_p.append(mp.Process(target=driver, args=(cmd_chunk[i], i)))
        all_p[i].start()
    

