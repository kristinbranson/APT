#!/bin/bash

str="$*"
cmd="testAPTci ${str}"
LD_LIBRARY_PATH=/misc/local/matlab-2020b/bin/glnxa64/old_libcrypto:$LD_LIBRARY_PATH \
/misc/local/matlab-2020b/bin/matlab -batch \
"disp(pwd); addpath ..; APT.setpath; $cmd"
