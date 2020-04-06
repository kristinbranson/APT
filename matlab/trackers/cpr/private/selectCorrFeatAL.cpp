/*
  AL rewrite of selectCorrFeat1.c, X.P. Burgos-Artizzu, P.Perona and Piotr Dollar.
*/

#include <mex.h>
#include <math.h>

/*
  Compute the covariance between F(:,f) and scalar(:,s)
  
  dfFtrs: [NxF], de-meaned features
  scalar: [NxS]
  mu: mean of scalar(:,s)
  s: column of scalar of interest (0-based)
  f: column of dfFtrs of interest (0-based)
*/
double computeCov(double* dfFtrs, double* scalar, double mu,
                  int s, int f, int N) {
  double cov = 0;
  for (int i=0;i<N;i++) {
    double ds = scalar[i+s*N]-mu;
    double x = dfFtrs[i+f*N]*ds;
    cov = cov+x;    
  }
  
  return cov/double(N);
}

/*

  [use,maxCos] = selectCorrFeatAL(pTar,ftrs,type,stdFtrs,dfFtrs,scalar,stdSc,muSc)

  - AL 20160302: pTar not used for anything besides dimension

 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if ( nrhs!=8 ) mexErrMsgTxt("Eight input arguments required.");
  if ( nlhs>2 ) mexErrMsgTxt("Too many output arguments.");
  if ( !mxIsClass(prhs[0], "double") || !mxIsClass(prhs[1], "double")
       || !mxIsClass(prhs[2], "double") || !mxIsClass(prhs[3], "double")
       || !mxIsClass(prhs[4], "double") || !mxIsClass(prhs[5], "double")
       || !mxIsClass(prhs[6], "double") || !mxIsClass(prhs[7], "double") )
    mexErrMsgTxt("Input arrays are of incorrect type.");

  unsigned int N = mxGetM(prhs[0]);
  unsigned int D = mxGetN(prhs[0]);
  mxAssert(mxGetM(prhs[1])==N,"Unexpected input arg size.");
  unsigned int F = mxGetN(prhs[1]);
  mxAssert(mxIsScalar(prhs[2]),"Unexpected input arg size.");
  mxAssert(mxGetM(prhs[3])==F && mxGetN(prhs[3])==F,"Unexpected input arg size.");
  mxAssert(mxGetM(prhs[4])==N && mxGetN(prhs[4])==F,"Unexpected input arg size.");
  mxAssert(mxGetM(prhs[5])==N,"Unexpected input arg size.");
  unsigned int S = mxGetN(prhs[5]);
  mxAssert(mxGetM(prhs[6])==1 && mxGetN(prhs[6])==S,"Unexpected input arg size.");
  mxAssert(mxGetM(prhs[7])==1 && mxGetN(prhs[7])==S,"Unexpected input arg size.");
  double *ysTar = (double*) mxGetData(prhs[0]); /* N x D, Target shapes to fit */
  double *data = (double*) mxGetData(prhs[1]);  /* N x F, Feature matrix */
  int type = (int) mxGetScalar(prhs[2]); /* 1=single-feature, 2=double-feature */
  double *stdFtrs  = (double*) mxGetData(prhs[3]); /* F x F, Std-of-(diff-between-features), see stdFtrs1.c */
  double *dfFtrs = (double*) mxGetData(prhs[4]); /* N x F, de-meaned features */
  double *scalar = (double*) mxGetData(prhs[5]); /* N x S, projections of ysTar along directions */
  double *stdSc = (double*) mxGetData(prhs[6]); /* 1 x S, std of scalar */
  double *muSc = (double*) mxGetData(prhs[7]); /* 1 x S, mean of scalar */
  
  plhs[0] = mxCreateNumericMatrix(type,S,mxDOUBLE_CLASS,mxREAL); /* type x S, features to use, 1-based indices */
  plhs[1] = mxCreateNumericMatrix(1,S,mxDOUBLE_CLASS,mxREAL); /* 1xS, maximum(abs(correlation)) for selected features */
  double *use = (double*) mxGetData(plhs[0]);
  double *maxCos = (double*) mxGetData(plhs[1]);
    
  /*
    For each s, find features f1 and f2 which maximize correlation
    (or anticorrelation) of f1-f2 to scalar(:,s). AFter some
    manipulation this is equivalent to maximizing the absolute value
    of

    (cov(scalar(:,s),data(:,f1)) - cov(scalar(:,s),data(:,f2))) / (stdSc(s))*stdFtrs(f1,f2))
    
  */
  double *covF = (double*) mxCalloc(F,sizeof(double));
  int *used = (int*) mxCalloc(F*F,sizeof(int));
  // used(i,j) will keep running track of feature-pairs i/j that have been selected 
  for (int s=0;s<S;s++) {
    double stdScs = stdSc[s];
    double muScs = muSc[s];
          
    for (int f=0;f<F;f++) {
      covF[f] = computeCov(dfFtrs,scalar,muScs,s,f,N);
    }  
          
    double valBest = -1; // best value of abs(correlation) encountered so far
    int f1Best = -1; // f1 index for best value so far
    int f2Best = -1; // etc
    for (int f1=0;f1<F;f1++) {
      for (int f2=f1+1;f2<F;f2++) {
        if (!used[f1+f2*F]) {
          double val = (covF[f1]-covF[f2])/(stdFtrs[f1+f2*F]*stdScs);
          val = fabs(val);
          // negative correlation just as good as positive; note also we only search for f2>f1
          
          if (val>valBest) { // Should automatically be true if
                             // f1Best, f2Best are still -1 (first
                             // viable f1,f2 encountered)
            f1Best = f1;
            f2Best = f2;
            valBest = val;
          }
        }
      }
    }
    mxAssert(f1Best!=-1 && f2Best!=-1,"No features chosen.");
    use[0+s*type] = f1Best+1; // returned indices are 1-based
    use[1+s*type] = f2Best+1;
    used[f1Best+f2Best*F] = 1;
    maxCos[s] = valBest;
  }

  mxFree(covF);
  mxFree(used);
}
