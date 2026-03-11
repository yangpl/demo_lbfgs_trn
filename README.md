# demo_lbfgs_trn

Small nonlinear-optimization demo for the 2D Rosenbrock problem.

Implemented methods:

- `method=0`: Newton-CG / truncated Newton
- `method=1`: L-BFGS
- `method=2`: nonlinear CG

## Build

```sh
make
```

## Run

```sh
./main
./main method=0 niter=100
./main method=1 npair=7
./main method=2 alpha=0.5
```

## API

Core callbacks:

- `optim_fg`: objective and gradient
- `optim_Hv`: Hessian-vector product

Main entry point:

- `optim_run(optim_t *opt, optim_fg fg, optim_Hv Hv)`

Usage by method:

- L-BFGS and nonlinear CG only need `fg`, so pass `Hv=NULL`
- Newton-CG needs both `fg` and `Hv`

## Parameters

- `method`: `0` Newton-CG, `1` L-BFGS, `2` nonlinear CG
- `niter`: maximum number of outer iterations
- `nls`: maximum number of line-search iterations
- `npair`: L-BFGS memory length
- `ncg`: maximum number of inner CG iterations for Newton-CG
- `tol`: relative gradient stopping tolerance
- `alpha`: initial line-search trial step
- `bound=1`: clip variables to `[0, 2]`
- `verb=1`: print progress and write `iterate.txt`

## Notes

- The default start point is `(1.5, 1.5)`.
- The Rosenbrock minimizer is `(1, 1)`.
- L-BFGS reaches the minimizer quickly on this demo.
- Newton-CG also converges to `(1, 1)`, but usually needs more outer iterations than L-BFGS. For this implementation, `method=0 niter=100` is a reasonable demo run.
- The iteration history is written to `iterate.txt` with fixed-width aligned columns.
