/* interface to implement l-BFGS and Newton-CG for nonlinear optimization */
#include "optim.h"

static bool optim_allocate_workspace(optim_t *opt)
{
  opt->x = alloc1float(opt->n);
  opt->g = alloc1float(opt->n);
  opt->d = alloc1float(opt->n);
  opt->xmin = alloc1float(opt->n);
  opt->xmax = alloc1float(opt->n);
  opt->g_prev = alloc1float(opt->n);
  opt->trial_x = alloc1float(opt->n);
  opt->trial_g = alloc1float(opt->n);
  opt->q = alloc1float(opt->n);
  opt->rho = alloc1float(opt->npair);
  opt->alp = alloc1float(opt->npair);
  opt->sk = alloc2float(opt->n, opt->npair);
  opt->yk = alloc2float(opt->n, opt->npair);

  return opt->x && opt->g && opt->d && opt->xmin && opt->xmax && opt->g_prev &&
         opt->trial_x && opt->trial_g && opt->q && opt->rho && opt->alp &&
         opt->sk && opt->yk;
}

static void optim_print_header(const optim_t *opt)
{
  if (!opt->verb) {
    return;
  }
  printf("method: %s\n", optim_method_name(opt->method));
  printf("dimension: %d\n", opt->n);
  printf("l-BFGS memory length: %d\n", opt->npair);
  printf("maximum iterations: %d\n", opt->niter);
  printf("gradient tolerance: %3.2e\n", opt->tol);
  printf("maximum line-search iterations: %d\n", opt->nls);
  printf("initial step length: alpha=%g\n", opt->alpha0);
}

static void optim_log_iteration(FILE *fp, optim_t *opt)
{
  if (!fp) {
    return;
  }
  fprintf(fp, "%6d %14.6e %14.6e %14.6e %10.4e %6d %8d\n",
          opt->iter, opt->fk, opt->fk / opt->f0, opt->gk_norm,
          opt->alpha, opt->ils, opt->igrad);
}

static void optim_choose_direction(optim_t *opt, optim_Hv Hv)
{
  int i;
  float beta_num, beta_den, beta;

  switch (opt->method) {
    case OPTIM_METHOD_NEWTON_CG:
      if (Hv) {
        cg_solve(opt->n, opt->x, opt->g, opt->d, Hv, opt);
      } else {
        flipsign(opt->n, opt->g, opt->d);
      }
      break;
    case OPTIM_METHOD_NLCG:
      if (opt->iter == 0) {
        flipsign(opt->n, opt->g, opt->d);
      } else {
        beta_num = dotprod(opt->n, opt->g, opt->g);
        beta_den = dotprod(opt->n, opt->g_prev, opt->g_prev);
        beta = (beta_den > 0.0f) ? beta_num / beta_den : 0.0f;
        for (i = 0; i < opt->n; ++i) {
          opt->d[i] = -opt->g[i] + beta * opt->d[i];
        }
      }
      memcpy(opt->g_prev, opt->g, opt->n * sizeof(float));
      break;
    case OPTIM_METHOD_LBFGS:
    default:
      if (opt->iter == 0) {
        flipsign(opt->n, opt->g, opt->d);
      } else {
        lbfgs_update(opt->n, opt->x, opt->g, opt->sk, opt->yk, opt);
        lbfgs_descent(opt->n, opt->g, opt->d, opt->sk, opt->yk, opt);
      }
      lbfgs_save(opt->n, opt->x, opt->g, opt->sk, opt->yk, opt);
      break;
  }
}

float l2norm(int n, const float *a)
{
  int i;
  float sum = 0.0f;

  for (i = 0; i < n; ++i) {
    sum += a[i] * a[i];
  }
  return sqrtf(sum);
}

float dotprod(int n, const float *a, const float *b)
{
  int i;
  float sum = 0.0f;

  for (i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

void flipsign(int n, const float *a, float *b)
{
  int i;
  for (i = 0; i < n; ++i) {
    b[i] = -a[i];
  }
}

bool lbfgs_pair_is_usable(int n, const float *s, const float *y)
{
  return dotprod(n, s, y) > 1e-12f;
}

bool optim_init(optim_t *opt, int n)
{
  int i;

  opt->n = n;
  if (opt->n <= 0 || opt->npair <= 0) {
    return false;
  }

  if (!optim_allocate_workspace(opt)) {
    optim_free(opt);
    return false;
  }

  for (i = 0; i < opt->n; ++i) {
    opt->xmin[i] = -FLT_MAX;
    opt->xmax[i] = FLT_MAX;
  }

  return true;
}

void optim_free(optim_t *opt)
{
  if (!opt) {
    return;
  }
  free1float(opt->x);
  free1float(opt->g);
  free1float(opt->d);
  free1float(opt->xmin);
  free1float(opt->xmax);
  free1float(opt->g_prev);
  free1float(opt->trial_x);
  free1float(opt->trial_g);
  free1float(opt->q);
  free1float(opt->rho);
  free1float(opt->alp);
  free2float(opt->sk);
  free2float(opt->yk);
  memset(opt, 0, sizeof(*opt));
}

const char *optim_method_name(int method)
{
  switch (method) {
    case OPTIM_METHOD_NEWTON_CG:
      return "Newton-CG";
    case OPTIM_METHOD_NLCG:
      return "Nonlinear CG";
    case OPTIM_METHOD_LBFGS:
    default:
      return "L-BFGS";
  }
}

void lbfgs_save(int n, const float *x, const float *g, float **sk, float **yk, optim_t *opt)
{
  int i;

  if (opt->kpair < opt->npair) {
    memcpy(sk[opt->kpair], x, n * sizeof(float));
    memcpy(yk[opt->kpair], g, n * sizeof(float));
    opt->kpair += 1;
    return;
  }

  for (i = 0; i < opt->npair - 1; ++i) {
    memcpy(sk[i], sk[i + 1], n * sizeof(float));
    memcpy(yk[i], yk[i + 1], n * sizeof(float));
  }
  memcpy(sk[opt->npair - 1], x, n * sizeof(float));
  memcpy(yk[opt->npair - 1], g, n * sizeof(float));
}

void lbfgs_update(int n, const float *x, const float *g, float **sk, float **yk, optim_t *opt)
{
  int i;
  int j = opt->kpair - 1;

  for (i = 0; i < n; ++i) {
    sk[j][i] = x[i] - sk[j][i];
    yk[j][i] = g[i] - yk[j][i];
  }

  if (!lbfgs_pair_is_usable(n, sk[j], yk[j])) {
    if (opt->verb) {
      printf("Discarding L-BFGS pair with nonpositive curvature.\n");
    }
    opt->kpair -= 1;
  }
}

void lbfgs_descent(int n, const float *g, float *d, float **sk, float **yk, optim_t *opt)
{
  int i, j;
  float sy, yy, beta, gamma;

  if (opt->kpair <= 0) {
    flipsign(n, g, d);
    return;
  }

  sy = dotprod(n, sk[opt->kpair - 1], yk[opt->kpair - 1]);
  yy = dotprod(n, yk[opt->kpair - 1], yk[opt->kpair - 1]);
  if (sy <= 1e-12f || yy <= 1e-12f) {
    flipsign(n, g, d);
    return;
  }

  memcpy(opt->q, g, n * sizeof(float));
  for (i = opt->kpair - 1; i >= 0; --i) {
    sy = dotprod(n, yk[i], sk[i]);
    if (sy <= 1e-12f) {
      flipsign(n, g, d);
      return;
    }
    opt->rho[i] = 1.0f / sy;
    opt->alp[i] = opt->rho[i] * dotprod(n, sk[i], opt->q);
    for (j = 0; j < n; ++j) {
      opt->q[j] -= opt->alp[i] * yk[i][j];
    }
  }

  gamma = dotprod(n, sk[opt->kpair - 1], yk[opt->kpair - 1]) /
          dotprod(n, yk[opt->kpair - 1], yk[opt->kpair - 1]);
  for (j = 0; j < n; ++j) {
    d[j] = gamma * opt->q[j];
  }

  for (i = 0; i < opt->kpair; ++i) {
    beta = opt->rho[i] * dotprod(n, yk[i], d);
    for (j = 0; j < n; ++j) {
      d[j] += (opt->alp[i] - beta) * sk[i][j];
    }
  }

  for (j = 0; j < n; ++j) {
    d[j] = -d[j];
  }
}

void boundx(float *x, int n, const float *xmin, const float *xmax)
{
  int i;
  for (i = 0; i < n; ++i) {
    if (x[i] < xmin[i]) {
      x[i] = xmin[i];
    }
    if (x[i] > xmax[i]) {
      x[i] = xmax[i];
    }
  }
}

void line_search(int n, float *x, float *g, float *d, optim_fg fg, optim_t *opt)
{
  int i;
  float alpha_lo = 0.0f;
  float alpha_hi = FLT_MAX;
  float step = (opt->alpha > 0.0f) ? opt->alpha : opt->alpha0;
  float fcost = opt->fk;
  float gxd0 = dotprod(n, g, d);
  float armijo_rhs;
  float gxd;

  memcpy(opt->trial_x, x, n * sizeof(float));
  memcpy(opt->trial_g, g, n * sizeof(float));

  if (gxd0 >= 0.0f) {
    if (opt->verb) {
      printf("Search direction is not descent. Falling back to steepest descent.\n");
    }
    flipsign(n, g, d);
    gxd0 = dotprod(n, g, d);
  }

  opt->ls_fail = 1;
  for (opt->ils = 0; opt->ils < opt->nls; ++opt->ils) {
    for (i = 0; i < n; ++i) {
      opt->trial_x[i] = x[i] + step * d[i];
    }
    if (opt->bound) {
      boundx(opt->trial_x, n, opt->xmin, opt->xmax);
    }

    fcost = fg(opt->trial_x, opt->trial_g);
    opt->igrad += 1;
    gxd = dotprod(n, opt->trial_g, d);
    armijo_rhs = opt->fk + opt->c1 * step * gxd0;

    if (fcost > armijo_rhs) {
      alpha_hi = step;
      step = 0.5f * (alpha_lo + alpha_hi);
    } else if (gxd < opt->c2 * gxd0) {
      alpha_lo = step;
      step = (alpha_hi < FLT_MAX) ? 0.5f * (alpha_lo + alpha_hi) : 10.0f * alpha_lo;
    } else {
      opt->ls_fail = 0;
      break;
    }

    if (step <= 0.0f || !isfinite(step)) {
      break;
    }
  }

  if (!opt->ls_fail) {
    opt->alpha = step;
    opt->fk = fcost;
    memcpy(x, opt->trial_x, n * sizeof(float));
    memcpy(g, opt->trial_g, n * sizeof(float));
    return;
  }

  opt->alpha = opt->alpha0;
}

void cg_solve(int n, const float *x, const float *g, float *d, optim_Hv Hv, optim_t *opt)
{
  int i, k;
  float rsold = 0.0f, rsnew = 0.0f, rs0, pAp, alpha, beta;
  float tol = 1e-3f;
  float *r = opt->trial_g;
  float *p = opt->trial_x;
  float *Ap = opt->q;

  memset(d, 0, n * sizeof(float));
  for (i = 0; i < n; ++i) {
    r[i] = -g[i];
    p[i] = r[i];
    rsold += r[i] * r[i];
  }

  rs0 = rsold;
  if (rs0 <= 0.0f) {
    flipsign(n, g, d);
    return;
  }

  for (k = 0; k < opt->ncg; ++k) {
    Hv(x, p, Ap);
    pAp = dotprod(n, p, Ap);
    if (pAp <= 1e-12f) {
      flipsign(n, g, d);
      return;
    }

    alpha = rsold / pAp;
    rsnew = 0.0f;
    for (i = 0; i < n; ++i) {
      d[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
      rsnew += r[i] * r[i];
    }

    if (opt->verb) {
      printf("===== CG %d, |r|^2=%e =====\n", k, rsnew);
    }
    if (rsnew < tol * rs0) {
      return;
    }

    beta = rsnew / rsold;
    for (i = 0; i < n; ++i) {
      p[i] = r[i] + beta * p[i];
    }
    rsold = rsnew;
  }
}

int optim_run(optim_t *opt, optim_fg fg, optim_Hv Hv)
{
  FILE *fp = NULL;

  if (!opt || !fg || !opt->x || !opt->g || !opt->d) {
    return OPTIM_STATUS_LINE_SEARCH_FAILED;
  }

  opt->igrad = 0;
  opt->iter = 0;
  opt->ils = 0;
  opt->kpair = 0;
  opt->ls_fail = 0;
  opt->status = OPTIM_STATUS_RUNNING;
  opt->alpha = (opt->alpha0 > 0.0f) ? opt->alpha0 : 1.0f;

  opt->f0 = fg(opt->x, opt->g);
  opt->fk = opt->f0;
  opt->igrad = 1;
  opt->g0_norm = l2norm(opt->n, opt->g);
  opt->gk_norm = opt->g0_norm;

  optim_print_header(opt);
  if (opt->verb) {
    fp = fopen("iterate.txt", "w");
    if (fp) {
      fprintf(fp, "================================================================================\n");
      fprintf(fp, "method: %s\n", optim_method_name(opt->method));
      fprintf(fp, "%6s %14s %14s %14s %10s %6s %8s\n",
              "iter", "fk", "fk/f0", "||gk||", "alpha", "nls", "ngrad");
      fprintf(fp, "================================================================================\n");
    }
  }

  for (opt->iter = 0; opt->iter < opt->niter; ++opt->iter) {
    opt->gk_norm = l2norm(opt->n, opt->g);
    if (opt->verb) {
      printf("iteration=%d fk=%g ||g||=%g\n", opt->iter, opt->fk, opt->gk_norm);
      optim_log_iteration(fp, opt);
    }

    if (opt->gk_norm <= opt->tol * MAX(1.0f, opt->g0_norm)) {
      opt->status = OPTIM_STATUS_CONVERGED;
      break;
    }

    optim_choose_direction(opt, Hv);
    line_search(opt->n, opt->x, opt->g, opt->d, fg, opt);
    if (opt->ls_fail) {
      opt->status = OPTIM_STATUS_LINE_SEARCH_FAILED;
      break;
    }
  }

  if (opt->status == OPTIM_STATUS_RUNNING) {
    opt->status = (opt->iter >= opt->niter) ? OPTIM_STATUS_MAX_ITER : OPTIM_STATUS_CONVERGED;
  }

  if (fp) {
    switch (opt->status) {
      case OPTIM_STATUS_CONVERGED:
        fprintf(fp, "==> Convergence reached.\n");
        break;
      case OPTIM_STATUS_MAX_ITER:
        fprintf(fp, "==> Maximum iteration number reached.\n");
        break;
      case OPTIM_STATUS_LINE_SEARCH_FAILED:
        fprintf(fp, "==> Line search failed.\n");
        break;
      default:
        break;
    }
    fclose(fp);
  }

  if (opt->verb) {
    int i;
    printf("final x:");
    for (i = 0; i < opt->n; ++i) {
      printf(" %g", opt->x[i]);
    }
    printf("\n");
  }

  return opt->status;
}
