#! /bin/bash

# Compare an output image to a target image using perceptual (visual)
# similarity, and pass/fail accordingly.  Uses ImageMagick (compare/identify),
# a system tool independent of the conda environment / image under test, so the
# check does not rely on whatever produced the output.
#
# Usage:
#   ./compare-to-target.bash <output-image> [target-image]
#
# The target image defaults to demo-output-target.jpg.  Exits nonzero (with a
# "FAIL:" message) if the output is missing, the wrong size, or too dissimilar.

set -e

output="$1"
target="${2:-demo-output-target.jpg}"
# Maximum PHASH (perceptual-hash) distance for the output to count as a match.
# 0 means visually identical; a harmless JPEG re-encode scores well under 0.1,
# while a genuinely different image scores several units.
threshold=1.0

if [ -z "$output" ] ; then
  echo "Usage: $0 <output-image> [target-image]"
  exit 1
fi

if ! command -v compare >/dev/null || ! command -v identify >/dev/null ; then
  echo "FAIL: ImageMagick (compare/identify) is required for the comparison but was not found"
  exit 1
fi

if [ ! -f "$output" ] ; then
  echo "FAIL: output image $output does not exist"
  exit 1
fi

if [ ! -f "$target" ] ; then
  echo "FAIL: target image $target does not exist"
  exit 1
fi

# The output must have the same dimensions as the target (PHASH is scale
# tolerant, so a dimension check is needed to catch wrong-size output).
outputDimensions=$(identify -format '%wx%h' "$output")
targetDimensions=$(identify -format '%wx%h' "$target")
if [ "$outputDimensions" != "$targetDimensions" ] ; then
  echo "FAIL: $output ($outputDimensions) and $target ($targetDimensions) have different dimensions"
  exit 1
fi

# Measure perceptual (visual) similarity.  compare exits 0/1 on a successful
# comparison (identical/different) and 2 on error, so treat 2 as failure.
set +e
score=$(compare -metric PHASH "$output" "$target" null: 2>&1)
compareStatus=$?
set -e
if [ "$compareStatus" -ge 2 ] ; then
  echo "FAIL: could not compare images: $score"
  exit 1
fi

isMatch=$(awk -v s="$score" -v t="$threshold" 'BEGIN { print (s <= t) ? 1 : 0 }')
if [ "$isMatch" -eq 1 ] ; then
  echo "PASS: $output matches $target (PHASH $score <= $threshold)"
else
  echo "FAIL: $output differs from $target (PHASH $score > $threshold)"
  exit 1
fi
