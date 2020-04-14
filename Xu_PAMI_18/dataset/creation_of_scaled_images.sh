#!/bin/bash

target="./Xu_PAMI_18/dataset/videos"
for f in "$target"/*
do
    echo $f
    filename="${f##*/}"
    filename="${filename%.*}"
    echo $filename
    mkdir "$target/$filename"
    ffmpeg -i "$f" -r 5 -vf scale=960:480 -q:v 1 -qmin 1 -qmax 1 "$target/$filename"/"%03d.jpg"
done
