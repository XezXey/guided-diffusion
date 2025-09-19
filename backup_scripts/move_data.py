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
    
    if '@' in args.target_path:
      is_dir = False
    else:
      is_dir = True
    # else:
      # logger.error("[#] --target_path need to be either directory or ssh machine")
      # exit()

    with open(args.json_file, "r") as f:
        data = json.load(f)
    
    for k, v in data.items():
        method = k
        if "img_dir" in v.keys():
          src_dir = v["img_dir"]
        elif "res_dir" in v.keys():
          src_dir = v["res_dir"]
        else:
          logger.error("[#] img_dir or res_dir key need to be exist.")
          exit()

        logger.info("#" * 100)
        logger.info(f"[#] Processing: {k}.")
        if not os.path.exists(src_dir):
            logger.error(f"[!] Path {src_dir} does not exist. Skipping.")
            continue
          
        dst_dir = f"{args.target_path}:{src_dir}" if not is_dir else f"{args.target_path}/{src_dir}"
        if is_dir and not os.path.exists(args.target_path):
          os.makedirs(dst_dir, exist_ok=True)
          cmd = ["rsync", "-azh", "--info=progress2", "--stats", src_dir, dst_dir]
        else:
          cmd = ["rsync", "-azh", "--mkpath", "--info=progress2", "--stats", src_dir, dst_dir]
          
        logger.info(f"[#] From: {src_dir}")
        logger.info(f"[#] To: {dst_dir}")
        logger.info(f"[#] Command: {' '.join(cmd)}")
        os.system(" ".join(cmd))
        logger.info("#" * 100)
