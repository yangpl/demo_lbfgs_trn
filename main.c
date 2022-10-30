/* template to implement l-BFGS algorithm */
/*
  Copyright (c) Pengliang Yang, 2017, Univ. Grenoble Alpes
  Homepage: https://yangpl.wordpress.com
  E-mail: ypl.2100@gmail.com  

  Reference: 
  [1] Numerical Optimization, Nocedal, 2nd edition, 2006 
  Algorithm 7.4 p. 178, Algorithm 7.5 p. 179   
  [2] https://en.wikipedia.org/wiki/Limited-memory_BFGS
  [3] SEISCOPE OPTIMIZATION toolbox
*/
#include <mpi.h>
#include <omp.h>
#include "cstd.h"
#include "lbfgs.h"

float rosenbrock_fg(float *x, float *g);
void rosenbrock_Hv(float *x, float *v, float *Hv);

int main (int argc, char **argv)
{
  int iproc, nproc, n, i;
  lbfgs_t *opt; //pointer for lbfgs_t parameters
  float fcost, beta, *g0;
  FILE *fp=NULL;

  // initialize MPI 
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);


  initargs(argc,argv);
  opt=alloc1(1, sizeof(lbfgs_t));
  
  /*-------------------------------------------------------------------------*/
  if(!getparint("niter", &opt->niter)) opt->niter = 100;/* maximum number of iterations */
  if(!getparint("nls", &opt->nls)) opt->nls = 20;/* maximum number of line searches */
  if(!getparfloat("tol", &opt->tol)) opt->tol = 1e-8;/* convergence tolerance */
  if(!getparint("npair", &opt->npair)) opt->npair = 5; /* l-BFGS memory length */
  if(!getparfloat("c1", &opt->c1)) opt->c1 = 1e-4; /* Nocedal value for Wolfe condition */
  if(!getparfloat("c2", &opt->c2)) opt->c2 = 0.9;  /* Nocedal value for Wolfe condition */
  if(!getparfloat("alpha", &opt->alpha)) opt->alpha = 1.;  /* initial step length */
  if(!getparint("bound", &opt->bound)) opt->bound = 0;/* 1 = bound on, 0 = off */
  if(!getparint("method", &opt->method)) opt->method = 2;//0=lBFGS; 1=Guass-Newton
  if(!getparint("ncg", &opt->ncg)) opt->ncg = 5;//Guass-Newton inversion
  
  //allocate 
  n=2;
  opt->x=alloc1float(n);
  opt->g=alloc1float(n);
  opt->d=alloc1float(n);
  opt->sk=alloc2float(n, opt->npair);
  opt->yk=alloc2float(n, opt->npair);
  opt->xmin=alloc1float(n);
  opt->xmax=alloc1float(n);
  for(i=0; i<n; i++){
    opt->xmin[i] = 0;
    opt->xmax[i] = 2.;
  }
  if(opt->method==2) g0 = alloc1float(n);

  //initialize
  opt->x[0]=1.5; 
  opt->x[1]=1.5;
  fcost = rosenbrock_fg(opt->x, opt->g);
  opt->f0=fcost;
  opt->fk=fcost;
  opt->igrad=0;
  opt->kpair=0;
  opt->ils=0;
  if(opt->verb){
    fp=fopen("iterate.txt","w");
    fprintf(fp,"==========================================================\n");
    fprintf(fp,"l-BFGS memory length: %d\n",opt->npair);
    fprintf(fp,"Maximum number of iterations: %d\n",opt->niter);
    fprintf(fp,"Convergence tolerance: %3.2e\n", opt->tol);
    fprintf(fp,"maximum number of line search: %d\n",opt->nls);
    fprintf(fp,"initial step length: alpha=%g\n",opt->alpha);
    fprintf(fp,"==========================================================\n");
    fprintf(fp,"iter    fk       fk/f0      ||gk||    alpha    nls   ngrad\n");
    fclose(fp);
  }

  //l-BFGS optimization 
  for(opt->iter=0; opt->iter< opt->niter; opt->iter++){
    if(opt->verb) printf("iteration=%d  fcost=%g\n", opt->iter,opt->fk/opt->f0);
    if(opt->verb){
      opt->gk_norm=l2norm(n, opt->g);
      fp=fopen("iterate.txt","a");
      fprintf(fp,"%3d   %3.2e  %3.2e   %3.2e  %3.2e  %3d  %4d\n",
	      opt->iter,opt->fk,opt->fk/opt->f0,opt->gk_norm,opt->alpha,opt->ils,opt->igrad);
      fclose(fp);

    }

    if(opt->method==0){//Newton-CG, solve Hv = -g 
      cg_solve(n, opt->x, opt->g, opt->d, rosenbrock_Hv, opt);

    }else if(opt->method==1){//l-BFGS
    
      if(opt->iter==0){//first iteration, no stored gradient
	flipsign(n, opt->g, opt->d);//descent direction=-gradient
      }else{
	lbfgs_update(n, opt->x, opt->g, opt->sk, opt->yk, opt);
	lbfgs_descent(n, opt->g, opt->d, opt->sk, opt->yk, opt);
      } 
      lbfgs_save(n, opt->x, opt->g, opt->sk, opt->yk, opt);

    }else if(opt->method==2){//NLCG
      if(opt->iter==0){
	for(i=0; i<n; i++) opt->d[i] = -opt->g[i];
      }else{
	beta = dotprod(n, opt->g, opt->g)/dotprod(n, g0, g0);
	for(i=0; i<n; i++) opt->d[i] = -opt->g[i] + beta*opt->d[i];
      }
      memcpy(g0, opt->g, n*sizeof(float));
      
    }
    line_search(n, opt->x, opt->g, opt->d, rosenbrock_fg, opt);

    if(opt->ls_fail){
      if(opt->verb){
	fp=fopen("iterate.txt","a");
	fprintf(fp, "==> Line search failed!\n");
	fclose(fp);
      }
      break;
    }
    if(opt->fk < opt->tol * opt->f0){
      if(opt->verb){
	fp=fopen("iterate.txt", "a");
	fprintf(fp, "==> Convergence reached!\n");
	fclose(fp);
      }
      break;
    }
  } 
  if(opt->verb) {
    if(opt->iter==opt->niter) {
      fp=fopen("iterate.txt","a");
      fprintf(fp, "==> Maximum iteration number reached!\n");
      fclose(fp);
    }
    printf("x[0]=%g  x[1]=%g \n", opt->x[0], opt->x[1]);
  }

  free1float(opt->x);
  free1float(opt->g);
  free1float(opt->d);
  free2float(opt->sk);
  free2float(opt->yk);
  free1(opt);
  if(opt->method==2) free1float(g0);
  
  MPI_Finalize();
  return 0;   
}
