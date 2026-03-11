#ifndef OPTIM_H
#define OPTIM_H

#include "cstd.h"

enum {
  OPTIM_METHOD_NEWTON_CG = 0,
  OPTIM_METHOD_LBFGS = 1,
  OPTIM_METHOD_NLCG = 2
};

enum {
  OPTIM_STATUS_RUNNING = 0,
  OPTIM_STATUS_CONVERGED = 1,
  OPTIM_STATUS_MAX_ITER = 2,
  OPTIM_STATUS_LINE_SEARCH_FAILED = 3
};

typedef float (*optim_fg)(const float *x, float *g);
typedef void (*optim_Hv)(const float *x, const float *v, float *Hv);

typedef struct {
  int n;
  int method;
  int niter;
  int nls;
  int npair;
  int ncg;
  int verb;
  int bound;

  int iter;
  int ils;
  int igrad;
  int kpair;
  int status;
  int ls_fail;

  float tol;
  float c1;
  float c2;
  float alpha;
  float alpha0;
  float f0;
  float fk;
  float g0_norm;
  float gk_norm;

  float *x;
  float *g;
  float *d;
  float *xmin;
  float *xmax;
  float *g_prev;
  float *trial_x;
  float *trial_g;
  float *q;
  float *rho;
  float *alp;
  float **sk;
  float **yk;
} optim_t;

float l2norm(int n, const float *a);
float dotprod(int n, const float *a, const float *b);
void flipsign(int n, const float *a, float *b);
bool lbfgs_pair_is_usable(int n, const float *s, const float *y);

void optim_config_defaults(optim_t *opt);
bool optim_init(optim_t *opt, int n);
void optim_free(optim_t *opt);
int optim_run(optim_t *opt, optim_fg fg, optim_Hv Hv);
const char *optim_method_name(int method);

void lbfgs_save(int n, const float *x, const float *grad, float **sk, float **yk, optim_t *opt);
void lbfgs_update(int n, const float *x, const float *grad, float **sk, float **yk, optim_t *opt);
void lbfgs_descent(int n, const float *grad, float *r, float **sk, float **yk, optim_t *opt);

void boundx(float *x, int n, const float *xmin, const float *xmax);
void line_search(int n, float *x, float *g, float *d, optim_fg fg, optim_t *opt);
void cg_solve(int n, const float *x, const float *g, float *d, optim_Hv Hv, optim_t *opt);

#endif
