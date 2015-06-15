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
typedef struct {
   double mu;
   double muSq;
} means;

means computeMeans(double *x, int N){
    int i;
    double xi;
    
    means mus;
    mus.mu=0;mus.muSq=0;
    for(i=0;i<N;i++){
        xi=x[i];
        mus.mu=mus.mu+xi;
        mus.muSq=mus.muSq+(xi*xi);
    }
    mus.mu=mus.mu/N; mus.muSq=mus.muSq/N;
    return mus;
}

double computeSTD(double *x, int N){
    double std;
    means mus;
    
    mus = computeMeans(x,N);
    std = sqrt(mus.muSq-(mus.mu*mus.mu));
    return std;
}

double computeDiffSTD(double* ftrs,int f1,int f2,int N){
    int i;
    double diffStd, *x;
    
    x = (double*) mxCalloc( N, sizeof(double) );
    for (i=0;i<N;i++){
        x[i]=ftrs[i+f1*N]-ftrs[i+f2*N];
    }
    diffStd = computeSTD(x,N);
    mxFree(x);
    return diffStd;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  int F,N,f1,f2;
  double *stdFtrs, *ftrs;
  
  /* Error checking on arguments */
  if( nrhs!=1) mexErrMsgTxt("Two input arguments required.");
  if( nlhs>1 ) mexErrMsgTxt("Too many output arguments.");
  if( !mxIsClass(prhs[0], "double"))
    mexErrMsgTxt("Input arrays are of incorrect type.");
  
  /* extract inputs */
  ftrs = (double*)   mxGetData(prhs[0]); /* N x F */
  N=mxGetM(prhs[0]); F=mxGetN(prhs[0]);
  
  plhs[0] = mxCreateNumericMatrix(F, F, mxDOUBLE_CLASS, mxREAL);
  stdFtrs = (double*) mxGetData(plhs[0]);
  
  for (f1=0;f1<F;f1++){
      for (f2=f1+1;f2<F;f2++){
        stdFtrs[f1+(f2*F)]=computeDiffSTD(ftrs,f1,f2,N);
      }
  } 
}