#! /bin/bash

set -e

# Send all of this script's output (and its subprocesses' stdout) to stderr, on
# a single stream.  That keeps the final pass/fail line from being interleaved
# with messages the tools print to stderr (e.g. deprecation warnings flushed at
# Python exit), so it reliably appears last.
exec 1>&2

output=demo-output.jpg

# Generate the output using the environment/image under test.
rm -f "$output"
python image_demo.py \
    demo-input.jpg \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --out-file "$output" \
    --draw-heatmap

# Compare the output to the target (host-side, via ImageMagick).
./compare-to-target.bash "$output"
