#!/bin/bash
path=$1
count=0

mkdir -p "$path/compare_rmv_border_order/"

for file in "$path/compare_rmv_border/frame[0-9]*.png"; do
  cp "$file" "$path/compare_rmv_border_order/frame$count.png"
  ((count++))
done