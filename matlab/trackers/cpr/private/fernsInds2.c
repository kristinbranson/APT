/*  Copyright 2013 X.P. Burgos-Artizzu, P.Perona and Piotr Dollar.  
  [xpburgos-at-gmail-dot-com]
 Please email me if you find bugs, or have suggestions or questions!
 Licensed under the Simplified BSD License [see bsd.txt]

  Please cite our paper if you use the code:
  Robust face landmark estimation under occlusion, 
  X.P. Burgos-Artizzu, P. Perona, P. Dollar (c)
  ICCV'13, Sydney, Australia  */

#include <mex.h>
#include <math.h>
typedef unsigned int uint;

double computeMean(double *x, int N, int d){
    int i;
    double mu;
    
    mu=0;
    for(i=0;i<N;i++){
        mu=mu+x[i+d*N];
    }
    return mu/N;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  int N, F, M, S, D, S2, n, f, m, s, d;
  double *data, *thrs, *ys, *dfYs, *sumys, *mu, *count;
  uint *fids, *inds;
  
  /* Error checking on arguments */
  if( nrhs!=4) mexErrMsgTxt("Four input arguments required.");
  if( nlhs>5 ) mexErrMsgTxt("Too many output arguments.");
  if( !mxIsClass(prhs[0], "double") || !mxIsClass(prhs[1], "uint32")
  || !mxIsClass(prhs[2], "double") || !mxIsClass(prhs[3], "double"))
    mexErrMsgTxt("Input arrays are of incorrect type.");
  
  /* extract inputs */
  data = (double*) mxGetData(prhs[0]); /* N x F */
  fids = (uint*)   mxGetData(prhs[1]); /* M x S */
  thrs = (double*) mxGetData(prhs[2]); /* N x F */
  ys = (double*) mxGetData(prhs[3]); /* N x D */
  N=mxGetM(prhs[0]); F=mxGetN(prhs[0]);
  M=mxGetM(prhs[1]); S=mxGetN(prhs[1]);
  D=mxGetN(prhs[3]);
  
  /* create outputs */
  plhs[0] = mxCreateNumericMatrix(N, M, mxUINT32_CLASS, mxREAL);
  inds = (uint*) mxGetData(plhs[0]); /* N x M */
  
  /* compute inds */
  for(m=0; m<M; m++) for(s=0; s<S; s++) for(n=0; n<N; n++) {
    inds[n+m*N]*=2; f=fids[m+s*M]-1;
    if( data[n+f*N]<thrs[m+s*M] ) inds[n+m*N]++;
  }
  
  plhs[4] = mxCreateNumericMatrix(N, D, mxDOUBLE_CLASS, mxREAL);
  dfYs = (double*) mxGetData(plhs[4]);
          
  plhs[1] = mxCreateNumericMatrix(1, D, mxDOUBLE_CLASS, mxREAL);
  mu = (double*) mxGetData(plhs[1]);
  
  for(d=0;d<D;d++){
      mu[d]=computeMean(ys,N,d);
      for(n=0;n<N;n++){
        dfYs[n+d*N] = ys[n+d*N]-mu[d];
      }
  }
       
  S2=pow(2,S);
  plhs[2] = mxCreateNumericMatrix(S2, D, mxDOUBLE_CLASS, mxREAL);
  sumys = (double*) mxGetData(plhs[2]); 
  
  plhs[3] = mxCreateNumericMatrix(S2, 1, mxDOUBLE_CLASS, mxREAL);
  count = (double*) mxGetData(plhs[3]); 
  
  for(n=0; n<N*M; n++){
      s=inds[n]; count[s]++;
      for(d=0;d<D;d++){
         sumys[s+d*S2]=sumys[s+d*S2]+dfYs[n+d*N];
      }
      inds[n]++;
  }
}
