#include <mex.h>
#include <math.h>

/*
  Compute the abs-covariance between dX(:,iF) and dB(:,iS), up to a normalization constant.
  
  dX: [NxF]
  dB: [NxS]
  Xsd: [F]
  Bsd: [S]  
  iF: 0-based index into cols of dX
  iS: 0-based index into cols of dB
*/
double computeCovVal(const double *dX,const double *dB,const double *Xsd,const double *Bsd,unsigned int N,int iF,int iS)
{
  double cov = 0.0;
  for (unsigned int i=0;i<N;i++) {
    double x = dX[i+iF*N];
    double b = dB[i+iS*N];
    cov += x*b; // we don't normalize by N
  }
  return fabs(cov) / Xsd[iF] / Bsd[iS]; // dividing by Bsd is actually unnec since we maximize for fixed iS
}
 

/*

  [use,maxVals] = selectFeatSingle(dX,Xsd,dB,Bsd)

    dX: [NxF] de-meaned features (column means of dX should be zero)
    Xsd: [1xF] column stds of X
    dB: [NxS] de-meaned projections of target shapes onto S directions
    Bsd: [1xS] column stds of B

    use: [S] selected features, col indices into dX (1-based indices)
    maxVals: [S] best/maximum correlation values associated with selected feats
  
  
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if ( nrhs!=4 ) mexErrMsgTxt("Four input arguments required.");
  if ( nlhs>2 ) mexErrMsgTxt("Too many output arguments.");
  if ( !mxIsClass(prhs[0], "double") || !mxIsClass(prhs[1], "double")
       || !mxIsClass(prhs[2], "double") || !mxIsClass(prhs[3], "double") )
    mexErrMsgTxt("Input arrays are of incorrect type."); 

  const mxArray *dX = prhs[0];
  const mxArray *Xsd = prhs[1];
  const mxArray *dB = prhs[2];
  const mxArray *Bsd = prhs[3];
  
  unsigned int N = mxGetM(dX);
  unsigned int F = mxGetN(dX);
  if (mxGetM(Xsd)!=1 || mxGetN(Xsd)!=F) mexErrMsgTxt("Invalid size Xsd.");
  if (mxGetM(dB)!=N) mexErrMsgTxt("Invalid size dB.");
  unsigned int S = mxGetN(dB);
  if (mxGetM(Bsd)!=1 || mxGetN(Bsd)!=S) mexErrMsgTxt("Invalid size Bsd.");

  const double *pdX = (const double*) mxGetData(dX);
  const double *pXsd = (const double*) mxGetData(Xsd);
  const double *pdB = (const double*) mxGetData(dB);
  const double *pBsd = (const double*) mxGetData(Bsd);
  
  plhs[0] = mxCreateNumericMatrix(1,S,mxDOUBLE_CLASS,mxREAL);
  plhs[1] = mxCreateNumericMatrix(1,S,mxDOUBLE_CLASS,mxREAL); 
  double *use = (double*) mxGetData(plhs[0]);
  double *maxVals = (double*) mxGetData(plhs[1]);   

  int *tfused = (int*) mxCalloc(F,sizeof(int));

  // tfused(i) will keep running track of features that have been previously selected
  for (int iS=0;iS<S;iS++) {
    double bestValSeen = 0.0;
    int bestValIF = -1; // 0-based index into tfused
    
    for (int iF=0;iF<F;iF++) {
      if (!tfused[iF]) {
        double covF = computeCovVal(pdX,pdB,pXsd,pBsd,N,iF,iS);
        if (covF>bestValSeen) {
          bestValSeen = covF;
          bestValIF = iF;
        }
      }
    }

    mxAssert(bestValIF > -1,"No feature selected");
    use[iS] = bestValIF+1; // 1-based
    maxVals[iS] = bestValSeen;
    mxAssert(!tfused[bestValIF],"foo");
    tfused[bestValIF] = 1;
  }

  mxFree(tfused);
}
