/*
  Rosenbrock function is: 
  f(x1,x2)=(1-x1)**2+100.*(x2-x1**2)**2    

  The gradient of the Rosenbrock function is:
  dx1f(x1,x2)= -2(x1-1)-400x1*(x2-x1**2)      
  dx2f(x1,x2)= 200(x2-x1**2)                  

  The Hessian operator of the Rosenbrok function is                                 
  dx1x1f(x1,x2)=-2-400(x2-x1**2)+800x1**2     
  dx1x2f(x1,x2)=-400x1                        
  dx2x2f(x1,x2)=200                           
*/
void rosenbrock_init(int n, float *x)
{
  x[0] = 1.5;
  x[1] = 1.5;
}

float rosenbrock_fg(float *x, float *g)
{
  /*
    The routine Rosenbrock returns  f(x1,x2) in fcost,
    (dx1f(x1,x2),dx2f(x1,x2)) in g for input parameter (x1,x2) in x           
  */
  float fcost;
  float tmp1, tmp2;

  tmp1=1.-x[0];
  tmp2=x[1]-x[0]*x[0];
  fcost=tmp1*tmp1 + 100.*tmp2*tmp2;
    
  g[0]=-2.*tmp1 - 400.*x[0]*tmp2;
  g[1]=200.*tmp2;

  return fcost;
}

void rosenbrock_Hv(float *x, float *v, float *Hv)
{
  /*
    The routine Rosenbrock_Hess returns Hessian-vector product H(x)d in output Hv  
    for input parameters x and d.
    H is the Hessian matrix; x=(x1,x2), d=(d1,d2) are two vector of R^2 
  */
  Hv[0]=(1200.*x[0]*x[0]-400.*x[1]+2.)*v[0] -400*x[0]*v[1];
  Hv[1]=-400.*x[0]*v[0]+200.*v[1];
}
