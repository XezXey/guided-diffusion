#!/bin/bash
read -p "What is your name? " name
echo $name
for word in $name
do
    echo $word
done
