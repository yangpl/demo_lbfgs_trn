#ifndef lbfgs_h
#define lbfgs_h

//============================================================
typedef struct {
    int niter; //total number of iterations
    int iter;
    float tol; //convergence tolerance
    int npair; //maximum number pair allowed, l-BFGS memory length 
    int kpair; //actual number of pair already stored
    int nls;   //maximum number of line search
    int ils;   //actual number of line search
    int igrad; //number of function and gradient evaluation
    float c1;  //Wolfe condition, Nocedal value=1e-4
    float c2;  //Wolfe condition, Nocedal value=0.9
    float alpha;//true step length, initial value=1
    float f0;  //initial misfit
    float fk;  //misfit at k-th iteration
    float gk_norm;
    float *x; //unknown parameter
    float *g; //gradient
    float *d; //descent direction
    float **sk, **yk;//storing vectors in two-loop recursion
    float *q, *alp, *rho; // parameters in two-loop recursion
    int bound;//1=clip x using upper and lower bounds,0=not
    float *xmin;//lower bound for x
    float *xmax;//upper bound for x
    int verb; //verbose information to print or not
    int preco;//precondition or not
  int ncg;
  int method;
    bool loop1;//flag to check 1st loop in two-loop recursion is done or not
    int ls_fail;
} optim_t;

typedef float (*optim_fg)(float*,float*); 
typedef void (*optim_Hv)(float*,float*,float *); 

float l2norm(int n, float *a);
/*< L2 norm of a vector >*/

float dotprod(int n, float *a, float *b);
/*< dot product of two vectors >*/

void flipsign(int n, float *a, float *b);
/*< reverse the sign of the vector >*/

void lbfgs_save(int n, float *x, float *grad, float **sk, float **yk, optim_t *opt);
/*< save current model and gradient >*/

void lbfgs_update(int n, float *x, float *grad, float **sk, float **yk, optim_t *opt);
/*< update current sk and yk >*/

void lbfgs_descent(int n, float *grad, float *r, float **sk, float **yk, optim_t *opt);
/*< calculate search direction (two-loop recursion) >*/

bool lbfgs_descent1(int n, float *g, float *q, float *rho, float *alp, 
		    float **sk, float **yk, optim_t *opt);
/*< calculate search direction (1st loop of the two-loop recursion) >*/

void lbfgs_descent2(int n, float *g, float *q, float *rho, float *alp, 
		    float **sk, float **yk, optim_t *opt);
/*< calculate search direction (2nd loop of the two-loop recursion) >*/

void boundx(float *x, int n, float *xmin, float *xmax);
/*< clip x based on lower and upper bounds >*/


void line_search(int n, //dimension of x
		float *x, //input vector x
		float *g, //gradient of misfit function
		float *d, //descent direction
		optim_fg fg, //subroutine to evaluation function and gradient
		optim_t *opt); //pointer of l-BFGS optimization parameters
/*< line search (Wolfe condition) >*/

void cg_solve(int n, //dimension of x
	      float *x, //input vector x
	      float *g, //gradient of misfit function
	      float *d, //descent direction
	      optim_Hv Hv, //subroutine to evaluation function and gradient
	      optim_t *opt); //pointer of l-BFGS optimization parameters

#endif
