%COMPILE ALL .c objective functions
disp('Compiling.......................................');
cd private
mex fernsInds2.c
mex maxCov1.c
mex selectCorrFeat1.c
mex stdFtrs1.c
cd ..
disp('DONE');