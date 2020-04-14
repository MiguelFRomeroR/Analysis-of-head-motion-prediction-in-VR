#!/bin/bash

target="/home/twipsy/PycharmProjects/UniformHeadMotionDataset/David_MMSys_18/dataset/Videos/Stimuli"
for f in "$target"/*
do
    if [[ $f =~ \.mp4$ ]]; then
        echo $f
        filename="${f##*/}"
        filename="${filename%.*}"
        echo $filename
        mkdir "$target/$filename"
        ffmpeg -i "$f" -r 5 -vf scale=960:480 -q:v 1 -qmin 1 -qmax 1 "$target/$filename"/"%03d.jpg"
    fi
done
