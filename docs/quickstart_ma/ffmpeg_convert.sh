#!/bin/bash

./ffmpeg2gif.sh APTStart.mp4 APTStart.gif "setpts=0.5*PTS,fps=10,scale=1920:-1"

./ffmpeg2gif.sh newproj.mp4 newproj.gif "setpts=0.5*PTS,fps=10,scale=1920:-1"

./ffmpeg2gif.sh LabelSequential.mp4 LabelSequential.gif "fps=20,scale=1920:-1"

./ffmpeg2gif.sh LabelMore.mp4 LabelMore.gif "setpts=2*PTS,fps=15,scale=1920:-1"

./ffmpeg2gif.sh LabelTemplateDrag.mp4 LabelTemplateDrag.gif "fps=10,scale=1920:-1"

./ffmpeg2gif.sh LabelTemplateClick.mp4 LabelTemplateClick.gif "fps=10,scale=1920:-1"

./ffmpeg2gif.sh LabelTemplateKeyboard.mp4 LabelTemplateKeyboard.gif "fps=10,scale=1920:-1"

./ffmpeg2gif.sh occluded_box.mp4 occluded_box.gif "setpts=2*PTS,fps=10,scale=1920:-1"

./ffmpeg2gif.sh BackendSetup.mp4 BackendSetup.gif "fps=10,scale=1920:-1"

./ffmpeg2gif.sh skeleton_out.mp4 skeleton_out.gif "setpts=2*PTS,fps=10,scale=1920:-1"

./ffmpeg2gif.sh relabel_updated.mp4 relabel_updated.gif "fps=10,scale=1920:-1"

