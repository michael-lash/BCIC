#include "mex.h"
#include "math.h" 
/*1/2\|x-y\|_2^2
/* c^T(x)_+<=d   (c is a positive vector)
   l<=x<=u. 
   l<=0<=u.
*/
void CheckInput(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /* Check for proper number of arguments. */
  if (nrhs < 1)
     mexErrMsgTxt("proj_truncsimplex(y,c,d,l,u,tol).\n");
  
  if(mxIsSparse(prhs[0]) || mxIsComplex(prhs[0]))
    mexErrMsgTxt("Input must be a dense real matrix\n");
}

double Evaluation(double* y, double* c, double* l, double* u, double lambda, mwSize p, int* activeId, int numactiveid)
{
       mwSize i;
       double sum;
       sum=0.0;
       for (i=0; i<numactiveid; ++i){
		   if (y[activeId[i]]-lambda*c[activeId[i]]>0){
				sum=sum+c[activeId[i]]*max(max(min(y[activeId[i]]-lambda*c[activeId[i]],u[activeId[i]]),l[activeId[i]]),0);
		   }
       }
       return sum;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
  double lambda,temp,lb,ub;
  double *y, *c, *d,*l,*u,*tol,*x,*finallambda,*finallb,*finalub;
  mwSize p, n, i;
  int *activeId;
  int numactiveid,count;
        
  CheckInput(nlhs, plhs, nrhs, prhs);
  
  p=mxGetM(prhs[0]);
  y=mxGetPr(prhs[0]);
  c=mxGetPr(prhs[1]);
  d=mxGetPr(prhs[2]);
  l=mxGetPr(prhs[3]);
  u=mxGetPr(prhs[4]);
  tol=mxGetPr(prhs[5]);

  plhs[0]=mxCreateDoubleMatrix(p,1,mxREAL);
  x=mxGetPr(plhs[0]);

  numactiveid=0;
  for (i=0; i<p; ++i){
	  if (y[i]>0){
		  numactiveid=numactiveid+1;
	  }
	  else{
		  x[i]=max(y[i],l[i]);
	  }
  } 
  activeId = (int*)malloc( numactiveid*sizeof(int) );
  count=0;
  for (i=0; i<p; ++i){
	  if (y[i]>0){
		  activeId[count]=i;
		  count=count+1;
	  }
  }

  /*plhs[1]=mxCreateDoubleMatrix(1,1,mxREAL);
  finallambda=mxGetPr(plhs[1]);
  plhs[2]=mxCreateDoubleMatrix(1,1,mxREAL);
  finallb=mxGetPr(plhs[2]);
  plhs[3]=mxCreateDoubleMatrix(1,1,mxREAL);
  finalub=mxGetPr(plhs[3]);*/
  
  ub=0.0;
  lb=0.0;
  temp=Evaluation(y, c, l, u, lb, p, activeId, numactiveid);
  if (temp<=d[0]){
      for (i=0; i<numactiveid; ++i){
		  if (y[activeId[i]]-lb*c[activeId[i]]>0){
			x[activeId[i]]=max(min(y[activeId[i]]-lb*c[activeId[i]],u[activeId[i]]),l[activeId[i]]);
		  }
		  else{
			x[activeId[i]]=0.0;
		  }
      } 
	return;
  }
  /* find a large lambda*/
  for (i=0; i<numactiveid; ++i){
      ub=max(ub, y[activeId[i]]/c[activeId[i]]);
  }
  temp=Evaluation(y, c, l, u, ub, p, activeId, numactiveid);
  if (temp>=d[0]){
      for (i=0; i<numactiveid; ++i){
		x[activeId[i]]=0.0;
      } 
   return;
  }
  /* bisection search*/
  lambda=(lb+ub)/2;
  temp=Evaluation(y, c, l, u, lambda, p, activeId, numactiveid);
  while(temp-d[0]>tol[0]||temp-d[0]<-tol[0]){
      if (temp-d[0]>0){
        lb=lambda;            
      }
      if (temp-d[0]<0){
        ub=lambda;            
      }
      lambda=(ub+lb)/2;
      temp=Evaluation(y, c, l, u, lambda, p, activeId, numactiveid);
  }
  for (i=0; i<numactiveid; ++i){
          if (y[activeId[i]]-lambda*c[activeId[i]]>0){
			x[activeId[i]]=max(min(y[activeId[i]]-lambda*c[activeId[i]],u[activeId[i]]),l[activeId[i]]);
		  }
		  else{
			x[activeId[i]]=0.0;
		  }
  } 
  //finallambda[0]=lambda;
  //finallb[0]=d[0];
  //finalub[0]=temp;
  return;
  
}
