#!/bin/bash
read -p "Input the sub_dataset to run (e.g. Anakin_2, Sull_1, Jon_1) : " srcname
read -p "Multiprocess chunk size : " chsize 

for s in $srcname
do
    mkdir -p ./final_videos/"$s"
    for dst in $(ls -a /data/mint/sampling/Videos_fulllight/log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml/ema_085000/valid/reverse_sampling/src="$s"/ | grep -v "^\.\{1,2\}$")
    do
        echo "[!] Running : "$s $dst
        echo -e "\n"
        echo "
            python unalign_parallel.py \
-i /data/mint/DPM_Dataset/Videos/"$s"/images/ \
-o /data/mint/videos_realigned/"$s"/"$s""$dst"/aligned/ \
-r /data/mint/sampling/Videos_fulllight/log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml/ema_085000/valid/reverse_sampling/src="$s"/"$dst"/diff=1000/ \
-c ./videos/"$s"/"$s""$dst"/composite/ \
-cmp ./videos/"$s"/"$s""$dst"/compare_rmv_border/ \
-chsize "$chsize" \
"
        python unalign_parallel.py \
            -i /data/mint/DPM_Dataset/Videos/"$s"/images/ \
            -o /data/mint/videos_realigned/"$s"/"$s""$dst"/aligned/ \
            -r /data/mint/sampling/Videos_fulllight/log=Masked_Face_woclip+BgNoHead+shadow_256_cfg=Masked_Face_woclip+BgNoHead+shadow_256.yaml/ema_085000/valid/reverse_sampling/src="$s"/"$dst"/diff=1000/ \
            -c ./videos/"$s"/"$s""$dst"/composite/ \
            -cmp ./videos/"$s"/"$s""$dst"/compare_rmv_border/ \
            -chsize "$chsize" \
            -ap_dir /data/mint/DPM_Dataset/Videos/"$s"/"$s"_align_params.npy
        echo "[!] Done : "$s


        # ffmpeg -y -i ./videos/"$s"/"$s""$dst"/compare_rmv_border/frame%d.png -c:v libx264 -pix_fmt yuv420p -crf 18 ./final_videos/"$s"/"$s""$dst".mp4
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

