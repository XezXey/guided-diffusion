#!/bin/bash
read -p "Input the sub_dataset to run (e.g. Anakin_2, Sull_1, Jon_1) : " srcname

for s in $srcname
do
    for dst in $(ls -a /data/mint/sampling/Videos_fulllight/log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml/ema_085000/valid/reverse_sampling/src="$s"/ | grep -v "^\.\{1,2\}$")
    do
            echo "[!] Running : "$s $dst
            echo -e "\n"

        ffmpeg -y -pattern_type glob -i ./videos/"$s"/"$s""$dst"/compare_rmv_border/frame*.png -c:v libx264 -pix_fmt yuv420p -crf 18 ./final_videos/"$s""$dst".mp4
        echo "[!] Done : "$s
    done
done
echo "[!] Finished all : "$srcname

# for d in $dname
# do
#     echo "[!] Running : "$d
#     echo -e "\n\n"

#     python unalign.py \
#         -i /data/mint/DPM_Dataset/Videos/"$d"/images/ \
#         -o /data/mint/videos_realigned/"$d"/aligned/ \
#         -r /data/mint/sampling/Videos/log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml/ema_085000/valid/reverse_sampling/src="$d"/light=66170/diff=1000/ \
#         -c ./videos/ironman_2_66170/composite/ \
#         -cmp ./videos/ironman_2_66170/compare_rmv_border/
#     echo "[!] Done : "$d
# done

# echo "[!] Finished all : "$dname

