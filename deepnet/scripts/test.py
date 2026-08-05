#! /usr/bin/env python3
"""Smoke-test the current environment/image by running the pose-estimation demo.

Runs ``image_demo.py`` to generate ``demo-output.jpg``, then compares it to the
known-good target with ``compare-to-target.py``.  Meant to be run inside the
environment/image under test (e.g. ``conda run --name <env> python test.py``, or
inside a container).

All output is sent to stderr on a single stream, so the final pass/fail line
from the comparison is not interleaved with messages the tools print to stderr
(e.g. deprecation warnings flushed at Python exit) and reliably appears last.

This script uses only the Python 3.6 standard library.
"""

import os
import subprocess
import sys


# The demo assets (all living in this scripts directory).
DEMO_SCRIPT = 'image_demo.py'
DEMO_INPUT_IMAGE = 'demo-input.jpg'
DEMO_CONFIG = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
DEMO_CHECKPOINT = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
COMPARE_SCRIPT = 'compare-to-target.py'
OUTPUT_IMAGE = 'demo-output.jpg'


def main():
  # Run the demo in this script's directory, then compare its output.
  scriptDirectory = os.path.dirname(os.path.abspath(__file__))
  os.chdir(scriptDirectory)

  # Generate the output using the environment/image under test.  Send child
  # stdout to stderr so everything stays on one stream (see the module docstring).
  if os.path.exists(OUTPUT_IMAGE):
    os.remove(OUTPUT_IMAGE)
  subprocess.run([sys.executable, DEMO_SCRIPT,
                  DEMO_INPUT_IMAGE, DEMO_CONFIG, DEMO_CHECKPOINT,
                  '--out-file', OUTPUT_IMAGE, '--draw-heatmap'],
                 stdout=sys.stderr,
                 check=True)

  # Compare the output to the target (host-side, via ImageMagick).
  subprocess.run([sys.executable, COMPARE_SCRIPT, OUTPUT_IMAGE],
                 stdout=sys.stderr,
                 check=True)


if __name__ == '__main__':
  try:
    main()
  except subprocess.CalledProcessError as calledProcessError:
    sys.stderr.write('Error: command failed (exit %s): %s\n'
                     % (calledProcessError.returncode, ' '.join(calledProcessError.cmd)))
    sys.exit(1)
