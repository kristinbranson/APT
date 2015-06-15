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

double computeMean(double *x, int N){
    int i;
    double mu;
    
    mu=0;
    for(i=0;i<N;i++){
        mu=mu+x[i];
    }
    return mu/N;
}

double computeCov(double* dfFtrs, double *dfSc, int f, int N){
    int i;
    double cov, *x;
    
    x = (double*) mxCalloc( N, sizeof(double) );
    for (i=0;i<N;i++){
        x[i]=dfFtrs[i+f*N]*dfSc[i];
    }
    cov = computeMean(x,N);
    mxFree(x);
    return cov;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  double *use,*covF,*dfFtrs,*dfSc,*stdFtrs,stdScM,maxCo,val;
  int N,F,f1,f2,f;
  
  /* Error checking on arguments */
  if( nrhs!=4) mexErrMsgTxt("Two input arguments required.");
  if( nlhs>1 ) mexErrMsgTxt("Too many output arguments.");
  if( !mxIsClass(prhs[0], "double") || !mxIsClass(prhs[1], "double")
  || !mxIsClass(prhs[2], "double"))
    mexErrMsgTxt("Input arrays are of incorrect type.");
  
  /* extract inputs */
  dfFtrs = (double*)   mxGetData(prhs[0]); /* N x F */
  dfSc = (double*)   mxGetData(prhs[1]); /* N x 1 */
  stdFtrs  = (double*)   mxGetData(prhs[2]); /* F x F */
  stdScM =   (double) mxGetScalar(prhs[3]); /*Scalar  */
  
  N=mxGetM(prhs[0]); F=mxGetN(prhs[0]); 
  
  covF = (double*) mxCalloc(F,sizeof(double)); 
  for (f=0;f<F;f++){  
       covF[f] = computeCov(dfFtrs,dfSc,f,N);
  }
  
  plhs[0] = mxCreateNumericMatrix(2, 1, mxDOUBLE_CLASS, mxREAL);
  use = (double*) mxGetData(plhs[0]);
  
  maxCo = 0; 
  for (f1=0;f1<F;f1++){
    for (f2=f1+1;f2<F;f2++){
        val=(covF[f1]-covF[f2])/(stdFtrs[f1+(f2*F)]*stdScM);
        if(val>maxCo){
            maxCo=val;use[0]=f1+1;use[1]=f2+1;
        }
    }
  }
  mxFree(covF);
}