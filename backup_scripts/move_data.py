import os, subprocess
import json
import guided_diffusion.mint_logger as mint_logger
logger = mint_logger.createLogger()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_path", required=True, type=str, help="Path to move data to")
parser.add_argument("--json_file", required=True, type=str, help="Path to JSON file that defines the data to move.")
args = parser.parse_args()

if __name__ == "__main__":
    logger.warning(f"[#] Moving data to {args.target_path}.")
    logger.warning(f"[#] Reading JSON file from {args.json_file}.")
    
    with open(args.json_file, "r") as f:
        data = json.load(f)
    
    for k, v in data.items():
        method = k
        src_dir = v["img_dir"]
        logger.info("#" * 100)
        logger.info(f"[#] Processing: {k}.")
        if not os.path.exists(src_dir):
            logger.error(f"[!] Path {src_dir} does not exist. Skipping.")
            continue
        dst_dir = f"{args.target_path}:{src_dir}"
        cmd = ["rsync", "-azh", "--mkpath", "--info=progress2", "--stats", src_dir, dst_dir]
        logger.info(f"[#] From: {src_dir}")
        logger.info(f"[#] To: {dst_dir}")
        logger.info(f"[#] Command: {' '.join(cmd)}")
        os.system(" ".join(cmd))
        logger.info("#" * 100)