#!/bin/bash
read -p "Input the sub_dataset to run (e.g. Anakin_2, Sull_1, Jon_1) : " dname
read -p "Chunk size : " chunk_size

# dir_path=/data/mint/DPM_Dataset/Videos/"$dname"/aligned_images/valid/
dir_path=/data/mint/DPM_Dataset/Videos/"$dname"/images/
# Variables
num_files=$(ls -1 "$dir_path" | wc -l)  # Count the number of files in the current directory
read -p "Enter start index [0]: " start_idx
read -p "Enter end index [$((num_files - 1))]: " end_idx
start_idx=$((start_idx < 0 ? 0 : start_idx))
end_idx=$((end_idx >= num_files ? num_files + 1 : end_idx))

start_indices=$(seq $start_idx $chunk_size $((end_idx - 1)))  # Generate a list of start indices
last_stop_idx=$((end_idx - 1))
stop_indices=$(seq $((start_idx + chunk_size)) $chunk_size $last_stop_idx; echo $end_idx)  # Generate a list of stop indices


# start_indices=$(seq 0 $chunk_size $((num_files - 1)))  # Generate a list of start indices
# last_stop_idx=$((num_files+1))
# stop_indices=$(seq $chunk_size $chunk_size $last_stop_idx; echo $last_stop_idx)  # Generate a list of stop indices

# Print the start and stop indices for each sub-chunk
for i in $(seq 1 $(echo "$start_indices" | wc -w)); 
do
    start=$(echo $start_indices | awk "{print \$${i}}")
    end=$(echo $stop_indices | awk "{print \$${i}}")
    end=$((end+1))
    echo "[#] Computing flow on Sub-chunk $i: $start - $end"
    echo "[!] Running : "$dname
    echo -e "\n\n"
    python ./compute_flow.py --image_dir "$dir_path" --output_savepath /data/mint/OptFlows/ --video_name "$dname" --config_opticalflow_path ./config_opticalflow.ini --chunk_size "$chunk_size" --idx "$start" "$end"
    echo "[!] Done : "$d
done

# for d in $dname
# do
# done

# echo "[!] Finished all : "$dname
                                                                                                                                                                                                                                                                                                                                                                                                                
