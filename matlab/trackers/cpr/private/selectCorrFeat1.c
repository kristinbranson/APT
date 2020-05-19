/*  Copyright 2013 X.P. Burgos-Artizzu, P.Perona and Piotr Dollar.  
  [xpburgos-at-gmail-dot-com]
 Please email me if you find bugs, or have suggestions or questions!
 Licensed under the Simplified BSD License [see bsd.txt]

  Please cite our paper if you use the code:
  Robust face landmark estimation under occlusion, 
  X.P. Burgos-Artizzu, P. Perona, P. Dollar (c)
  ICCV'13, Sydney, Australia  */

// Modified by Allen Lee, Kristin Branson

#include <mex.h>
#include <math.h>
typedef unsigned int uint;

/*
  dfFtrs: [NxF]
  scalar: [NxS]
  mu: mean of scalar(:,s)
  s: column of scalar to investigate
  f: column of dfFtrs etc
*/
double computeCov(double* dfFtrs, double* scalar, double mu,
        int s, int f, int N) {
    int i;
    double cov, x, df;
    
    cov=0;
    for (i=0;i<N;i++){
        df=scalar[i+s*N]-mu;
        x=dfFtrs[i+f*N]*df;
        cov=cov+x;    
    }
    
    return cov/N;
}

/*

  [use,maxCos] = selectCorrFeat1(pTar,ftrs,type,stdFtrs,dfFtrs,scalar,stdSc,muSc)

 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  double *use,*covF,*dfFtrs,*stdFtrs,maxCo,val,*ysTar,*maxCos;/**dfSc*/
  double *data, *stdSc, *muSc,*scalar,stdScs,muScs;
  int N,F,D,f1,f2,f,S,d,s,type,*used;
  
  /* Error checking on arguments */
  if( nrhs!=8 ) mexErrMsgTxt("Eight input arguments required.");
  if( nlhs>2 ) mexErrMsgTxt("Too many output arguments.");
  if( !mxIsClass(prhs[0], "double") || !mxIsClass(prhs[1], "double")
  || !mxIsClass(prhs[2], "double") || !mxIsClass(prhs[3], "double")
  || !mxIsClass(prhs[4], "double") || !mxIsClass(prhs[5], "double")
  || !mxIsClass(prhs[6], "double") || !mxIsClass(prhs[7], "double"))
    mexErrMsgTxt("Input arrays are of incorrect type.");
  
  /* extract inputs */
  ysTar = (double*) mxGetData(prhs[0]); /* N x D, Target shapes to fit */ 
  data = (double*) mxGetData(prhs[1]);  /* N x F, Feature matrix */
  type = (int) mxGetScalar(prhs[2]); /* 1=single-feature, 2=double-feature */
  stdFtrs  = (double*) mxGetData(prhs[3]); /* F x F, Std-of-(diff-between-features), see stdFtrs1.c  */
  dfFtrs = (double*) mxGetData(prhs[4]); /* N x F, de-meaned features */
  scalar = (double*) mxGetData(prhs[5]); /* N x S, projections of ysTar along directions */
  stdSc = (double*) mxGetData(prhs[6]); /* 1 x S, std of scalar */
  muSc = (double*) mxGetData(prhs[7]); /* 1 x S, mean of scalar */
  /*group = (double*) mxGetData(prhs[8]); N x F */
  /*exclude = (double*) mxGetData(prhs[9]); 1 x 1 */
  
  N=mxGetM(prhs[0]); D=mxGetN(prhs[0]); 
  F=mxGetN(prhs[1]); S=mxGetN(prhs[5]); 
  
  plhs[0] = mxCreateNumericMatrix(type, S, mxDOUBLE_CLASS, mxREAL); /* typexS, features to use */
  plhs[1] = mxCreateNumericMatrix(1, S, mxDOUBLE_CLASS, mxREAL);
  use = (double*) mxGetData(plhs[0]);
  maxCos = (double*) mxGetData(plhs[1]);
    
  covF = (double*) mxCalloc(F,sizeof(double));
  
  used = (int*) mxCalloc(F*F,sizeof(int));
  
  for (s=0;s<S;s++){
          stdScs=stdSc[s];muScs=muSc[s];
          
          for (f=0;f<F;f++){
              covF[f] = computeCov(dfFtrs,scalar,muScs,s,f,N);
          }  
          
          maxCo = 0;
          for (f1=0;f1<F;f1++){
                for (f2=f1+1;f2<F;f2++){
                  if(!used[f1*F+f2]){
                    val=(covF[f1]-covF[f2])/(stdFtrs[f1+(f2*F)]*stdScs);
                    if(f1==0 && f2==1){
                        maxCo=val;use[s*type]=1;use[1+s*type]=2;
                        used[f1*F+f2]=1;
                        maxCos[s]=maxCo;
                    }
                    else if(val>maxCo){
                        maxCo=val;use[s*type]=f1+1;use[1+s*type]=f2+1;
                        used[f1*F+f2]=1;
                        maxCos[s]=maxCo;
                    }
                   }
              }
        }
  }
  mxFree(covF);mxFree(used);
}
