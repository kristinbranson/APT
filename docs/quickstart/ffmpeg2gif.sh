#!/bin/sh

#ffmpeg2gif input.mp4 output.gif "fps=15,scale=1920"
#copied from https://cassidy.codes/blog/2017/04/25/ffmpeg-frames-to-gif-optimization/

palette="/tmp/palette.png"

#filters="fps=15,scale=320:-1:flags=lanczos"
filters="$3:flags=lanczos"

echo $filters

ffmpeg -v warning -i $1 -vf "$filters,palettegen=stats_mode=diff" -y $palette

ffmpeg -i $1 -i $palette -lavfi "$filters,paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" -y $2
