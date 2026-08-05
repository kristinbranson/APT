#! /usr/bin/env python3
"""Compare an output image to a target image by perceptual (visual) similarity.

Uses ImageMagick (``compare``/``identify``), a system tool independent of the
conda environment / image under test, so the check does not rely on whatever
produced the output.

Usage:
    ./compare-to-target.py <output-image> [target-image]

The target image defaults to ``demo-output-target.jpg``.  Exits nonzero (with a
"FAIL:" message) if the output is missing, the wrong size, or too dissimilar.

This script uses only the Python 3.6 standard library.
"""

import os
import shutil
import subprocess
import sys


# Maximum PHASH (perceptual-hash) distance for the output to count as a match.
# 0 means visually identical; a harmless JPEG re-encode scores well under 0.1,
# while a genuinely different image scores several units.
THRESHOLD = 1.0

# Default target image to compare against.
DEFAULT_TARGET = 'demo-output-target.jpg'


def fail(message):
  # Print a FAIL message and exit nonzero.
  print('FAIL: %s' % message)
  sys.exit(1)


def identify_dimensions(imagePath):
  # Return the "<width>x<height>" string for an image, via ImageMagick identify.
  completedProcess = subprocess.run(['identify', '-format', '%wx%h', imagePath],
                                    stdout=subprocess.PIPE,
                                    universal_newlines=True,
                                    check=True)
  return completedProcess.stdout.strip()


def phash_distance(outputPath, targetPath):
  # Return (status, distanceText) from ImageMagick compare using the PHASH
  # metric.  compare writes the metric to stderr and exits 0/1 on a successful
  # comparison (identical/different) and 2 on error.
  completedProcess = subprocess.run(['compare', '-metric', 'PHASH', outputPath, targetPath, 'null:'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True)
  return completedProcess.returncode, completedProcess.stdout.strip()


def main():
  # Parse arguments and pass/fail the comparison.
  if len(sys.argv) < 2 or not sys.argv[1]:
    print('Usage: %s <output-image> [target-image]' % os.path.basename(sys.argv[0]))
    sys.exit(1)
  outputPath = sys.argv[1]
  targetPath = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TARGET

  if shutil.which('compare') is None or shutil.which('identify') is None:
    fail('ImageMagick (compare/identify) is required for the comparison but was not found')

  if not os.path.isfile(outputPath):
    fail('output image %s does not exist' % outputPath)
  if not os.path.isfile(targetPath):
    fail('target image %s does not exist' % targetPath)

  # The output must have the same dimensions as the target (PHASH is scale
  # tolerant, so a dimension check is needed to catch wrong-size output).
  outputDimensions = identify_dimensions(outputPath)
  targetDimensions = identify_dimensions(targetPath)
  if outputDimensions != targetDimensions:
    fail('%s (%s) and %s (%s) have different dimensions'
         % (outputPath, outputDimensions, targetPath, targetDimensions))

  # Measure perceptual (visual) similarity.
  compareStatus, distanceText = phash_distance(outputPath, targetPath)
  if compareStatus >= 2:
    fail('could not compare images: %s' % distanceText)

  try:
    distance = float(distanceText.split()[0])
  except (ValueError, IndexError):
    fail('could not parse PHASH distance from: %s' % distanceText)

  if distance <= THRESHOLD:
    print('PASS: %s matches %s (PHASH %s <= %s)' % (outputPath, targetPath, distanceText, THRESHOLD))
  else:
    fail('%s differs from %s (PHASH %s > %s)' % (outputPath, targetPath, distanceText, THRESHOLD))


if __name__ == '__main__':
  try:
    main()
  except subprocess.CalledProcessError as calledProcessError:
    fail('command failed (exit %s): %s'
         % (calledProcessError.returncode, ' '.join(calledProcessError.cmd)))
