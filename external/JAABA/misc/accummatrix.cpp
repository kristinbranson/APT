#include "mex.h"
#include <string.h>
#include <stdio.h>

#define IDX 0
#define W 1
#define NBINS 2

void accummatrix(unsigned char *idx0, double *w0, double *counts0, 
		 int D, int N, int nbins){

  // indices
  int d = 0; 
  int n = 0;
  int i = 0;

  // pointers
  unsigned char *idx;
  double *counts;
  double *w;
  idx = idx0; 
  counts = counts0;
  w = w0;

  memset(counts,0,D*nbins*sizeof(double));

  // loop over dimensions
  for(d=0; d<D; d++,counts+=nbins){

    for(n=0,w=w0; n<N; n++,idx++,w++){
      counts[(*idx)-1] += *w;
    }
  }
}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[]){

  // main inputs
  unsigned char *idx;
  double *w;
  int D;
  int N;
  int nbins;

  // output
  double *counts;

  // check arguments
  if( (nrhs != 3) || (nlhs>1) ){
    mexErrMsgTxt("Usage: counts = accummatrix(idx,w,nbins)");
    return;
  }


  // check format of arguments
  if( !mxIsDouble(prhs[W]) ){
    mexErrMsgTxt("w must be of class double.");
    return;
  }
  if( !mxIsUint8(prhs[IDX]) ){
    mexErrMsgTxt("idx must be of class uint8.");
    return;
  }
  if( !mxIsNumeric(prhs[NBINS]) ){
    mexErrMsgTxt("nbins must be a number.");
    return;
  }

  // get input sizes
  N = (int)mxGetM(prhs[IDX]);
  D = (int)mxGetN(prhs[IDX]);

  // make sure input sizes match
  if(N != (int)mxGetM(prhs[W])){
    mexErrMsgTxt("Number of rows in idx must match number of rows in w");
    return;
  }

  if(1 != mxGetN(prhs[W])){
    mexErrMsgTxt("w must be a column vector");
    return;
  }

  // make sure input sizes match
  if( (1 != mxGetM(prhs[NBINS])) || (1 != mxGetN(prhs[NBINS])) ){
    mexErrMsgTxt("nbins must be a scalar");
    return;
  }

  // get inputs
  idx = (unsigned char*)mxGetData(prhs[IDX]);
  w = mxGetPr(prhs[W]);

  nbins = (int)mxGetScalar(prhs[NBINS]);

  // create output
  plhs[0] = mxCreateDoubleMatrix(nbins,D,mxREAL);
  counts = mxGetPr(plhs[0]);

  accummatrix(idx,w,counts,D,N,nbins);

}
