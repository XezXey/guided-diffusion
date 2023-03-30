#!/bin/bash
read -p "Input the video(s) to run (e.g. Anakin_2, Sull_1, Jon_1) : " videos

for vid_name in $videos
do
    echo "[!] Running : "$vid_name
    echo -e "\n\n"
    python ./align.py -i /data/mint/DPM_Dataset/Videos/"$vid_name"/images -o /data/mint/tmp/"$vid_name"/aligned_images/valid/  
    echo "[!] Done : "$d
done

echo "[!] Finished all : "$videos
                                                                                                                                                                                                                                                                                                                                                                                                                