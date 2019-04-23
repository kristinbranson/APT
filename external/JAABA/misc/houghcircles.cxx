#include "mex.h"
#include <stdio.h>
#include <math.h>

#define SQUARED(XX) ((XX)*(XX))

int binround(double x, double * edges, int nbins);

void houghcircles_accumulate(double* acc,
			     double * x, double * y,
			     double* bina, double* binb, double * binr,
			     int npts, int na, int nb,
			     int nr){
  
  int ib, ir;
  double rsquared, xpt, ypt;
  int pt, iacc;
  double d;
  int soln1, soln2;
  double *binrsquared;
  binrsquared = new double[nr];
  for(ir = 0; ir < nr; ir++){
    binrsquared[ir] = binr[ir]*binr[ir];
  }

  for(pt = 0; pt < npts; pt++){
    iacc = 0;
    xpt = x[pt];
    ypt = y[pt];
    for(ir = 0; ir < nr; ir++){
      rsquared = binrsquared[ir];
      for(ib = 0; ib < nb; ib++,iacc+=na){

	// solve for a such that (x - a)^2 + (y - b)^2 = r^2
	// x^2 - 2ax + a^2 + (y - b)^2 - r^2 = 0
	// a^2 - 2ax + (x^2 + (y - b)^2 - r^2) = 0
	// a = x +/- sqrt(x^2 - (x^2 + (y - b)^2 - r^2))
	//   = x +/- sqrt( r^2 - (y - b)^2 )
	// in order for there to be a solution for this b, we 
	// require r >= abs(y - b)

	d = SQUARED(ypt - binb[ib]);
	if( d > rsquared){
	  // no solution
	  continue;
	}
	d = sqrt(rsquared - d);
	soln1 = binround(xpt+d,bina,na);
	soln2 = binround(xpt-d,bina,na);
	
	// one solution
	if(soln1 == soln2){
	  // out of bounds?
	  if(soln1 < 0){
	    continue;
	  }
	  acc[iacc+soln1]++;
	  continue;
	}

	// two solutions 
	if(soln1 >= 0){
	  acc[iacc+soln1]++;
	}
	if(soln2 >= 0){
	  acc[iacc+soln2]++;
	}

      }
    }
    
  }
  
  delete binrsquared;
  rsquared = 0;

}

int binround(double x, double * edges, int nbins){

  int bin;

  // check for out of bounds
  if( (x < edges[0]) || (x >= edges[nbins]) ){
    return(-1);
  }

  // for now, just do a linear search
  for(bin = 1; bin <= nbins; bin++){
    if( x < edges[bin] ){
      return(bin-1);
    }
  } 
  return(-1);

}

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[]){

  double * acc, *bina, *binb, *binr, *x, *y;
  int npts, na, nb, nr, i;
  int dims[3];

  if( (nrhs!=5) || (nlhs>1) )
    mexErrMsgTxt("Usage: acc = houghcircles(x,y,binedgesa,bincentersb,bincentersr)");

  /* check format of arguments */
  for(i = 0; i < 5; i++){
    if( !mxIsDouble(prhs[i]) || mxIsComplex(prhs[i]) )
      mexErrMsgTxt("All inputs must be real doubles.");
    }
  
  /* get sizes of inputs */
  npts = mxGetM(prhs[0]);
  if(npts == 1) npts = mxGetN(prhs[0]);
  if( !( ((npts == mxGetM(prhs[1])) && (1 == mxGetN(prhs[1]))) ||
	 ((npts == mxGetN(prhs[1])) && (1 == mxGetM(prhs[1]))) ) ){
    mexErrMsgTxt("Inputs x and y must be the same size.");
  }
  na = mxGetM(prhs[2]);
  if(na == 1) na = mxGetN(prhs[2]);
  if( na < 2 ){
    mexErrMsgTxt("Input a must be at least length 2.");
  }
  na--;
  nb = mxGetM(prhs[3]);
  if(nb == 1) nb = mxGetN(prhs[3]);
  nr = mxGetM(prhs[4]);
  if(nr == 1) nr = mxGetN(prhs[4]);

  x = mxGetPr(prhs[0]);
  y = mxGetPr(prhs[1]);
  bina = mxGetPr(prhs[2]);
  binb = mxGetPr(prhs[3]);
  binr = mxGetPr(prhs[4]);

  /*  set the output pointer to the output matrix */
  dims[0] = na; dims[1] = nb; dims[2] = nr;
  plhs[0] = mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);

  /*  create a C pointer to a copy of the output matrix */
  acc = mxGetPr(plhs[0]);
  for(i = 0; i < na*nb*nr; i++) acc[i] = 0;

  houghcircles_accumulate(acc,x,y,bina,binb,binr,npts,na,nb,nr);

}
