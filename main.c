/* driver for nonlinear optimization demos */
#include "optim.h"

void rosenbrock_init(int n, float *x);
float rosenbrock_fg(const float *x, float *g);
void rosenbrock_Hv(const float *x, const float *v, float *Hv);


int main(int argc, char **argv)
{
  int i;
  optim_t opt;

  initargs(argc, argv);
  if(!getparint("verb", &opt.verb)) opt.verb = 1;
  if(!getparint("niter", &opt.niter)) opt.niter = 100;
  if(!getparint("nls", &opt.nls)) opt.nls = 20;
  if(!getparint("npair", &opt.npair)) opt.npair = 5;
  if(!getparint("bound", &opt.bound)) opt.bound = 0;
  if(!getparint("method", &opt.method)) opt.method = 1;//0=NewtonCG;1=LBFGS;2=NLCG
  if(!getparint("ncg", &opt.ncg)) opt.ncg = 5;
  if(!getparfloat("tol", &opt.tol)) opt.tol = 1e-6f;
  if(!getparfloat("c1", &opt.c1)) opt.c1 = 1e-4f;
  if(!getparfloat("c2", &opt.c2)) opt.c2 = 0.9f;
  if(!getparfloat("alpha", &opt.alpha0)) opt.alpha0 = 1.0f;
  if(opt.alpha0 <= 0.0f) opt.alpha0 = 1.0f;
  opt.alpha = opt.alpha0;
  if (!optim_init(&opt, 2)) {
    fprintf(stderr, "failed to initialize optimizer\n");
    return EXIT_FAILURE;
  }

  if (opt.npair <= 0) {
    fprintf(stderr, "npair must be positive\n");
    optim_free(&opt);
    return EXIT_FAILURE;
  }

  if (opt.bound) {
    for (i = 0; i < opt.n; ++i) {
      opt.xmin[i] = 0.0f;
      opt.xmax[i] = 2.0f;
    }
  }

  rosenbrock_init(opt.n, opt.x);
  if (opt.method == OPTIM_METHOD_NEWTON_CG) {
    if (optim_run(&opt, rosenbrock_fg, rosenbrock_Hv) == OPTIM_STATUS_LINE_SEARCH_FAILED) {
      optim_free(&opt);
      return EXIT_FAILURE;
    }
  } else {
    if (optim_run(&opt, rosenbrock_fg, NULL) == OPTIM_STATUS_LINE_SEARCH_FAILED) {
      optim_free(&opt);
      return EXIT_FAILURE;
    }
  }

  optim_free(&opt);
  return EXIT_SUCCESS;
}
