r"""Build the NNGP lookup grid (Lee et al. 2018, Section 2.5) for any activation.

Why this file exists
--------------------
nngp.NNGPKernel needs, for each activation phi, a precomputed table of

    q_aa(v)    = E_{z ~ N(0, v)}[ phi(z)^2 ]                       (variance map)
    q_ab(v, c) = E_{(z1,z2) ~ N(0, v [[1, c], [c, 1]])}[ phi(z1) phi(z2) ]

on a grid of pre-activation variances v and correlations c.  This is exactly
F_phi in the kernel recursion (Section 2.3, Eq. 4 & 5); the paper's Section
2.5 explains the lookup table + bilinear interpolation.  The repo ships the
tables only for tanh and ReLU (grid_data/).

The repo *can* build a new table itself (nngp._compute_qmap_grid), but it
materialises a [n_gauss, n_gauss, n_corr] float64 tensor (~1 GB) for every
variance value and runs one per CPU core in parallel.  On a normal laptop
inside Docker that runs out of memory.  This script computes the *same
quantities with the same discretisation* in NumPy, one small chunk at a time,
and writes the file in the exact format nngp.py loads.

Faithfulness check: `--validate` regenerates rows of the shipped tanh / ReLU
grids and reports the max absolute difference (it should be ~1e-12), and also
checks ReLU against its known closed form (the arc-cosine kernel of Cho & Saul
2009, which the paper cites for ReLU).

Usage (inside the container, from /nngp):
    python make_grid.py --nonlinearity=gelu            # writes grid_data/...
    python make_grid.py --validate                     # correctness checks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing
import os
import time

import numpy as np
import math

# Same default grid as run_experiments.py / uncertainty_plot.py flags:
# n_gauss=501, n_var=501, n_corr=500, max_var=100, max_gauss=10.
N_GAUSS, N_VAR, N_CORR, MAX_VAR, MAX_GAUSS = 501, 501, 500, 100, 10
MIN_VAR, MAX_CORR = 1e-8, 0.99999  # defaults of nngp._compute_qmap_grid


_erf = np.vectorize(math.erf)  # avoids a scipy dependency; z has 501 points


def _np_gelu(x):
  return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


NP_ACTIVATIONS = {
    'tanh': np.tanh,
    'relu': lambda x: np.maximum(x, 0.0),
    'gelu': _np_gelu,
}


def grid_axes(n_gauss=N_GAUSS, n_var=N_VAR, n_corr=N_CORR,
              max_var=MAX_VAR, max_gauss=MAX_GAUSS):
  """Mirror of the linspace calls in nngp._compute_qmap_grid."""
  z = np.linspace(-max_gauss, max_gauss, n_gauss)
  var_aa = np.linspace(MIN_VAR, max_var, n_var)
  corr_ab = np.linspace(-MAX_CORR, MAX_CORR, n_corr)
  return z, var_aa, corr_ab


def qaa_row(phi_z, z, var):
  """q_aa(v): Gaussian-weighted average of phi(z)^2 on the fixed z grid.

  Same as nngp.py: weights are exp(-z^2 / 2v), normalised to sum to 1
  (a discrete version of the N(0, v) density).
  """
  logw = -0.5 * z**2 / var
  w = np.exp(logw - logw.max())
  w /= w.sum()
  return np.sum(phi_z**2 * w)


def qab_row(args):
  """One row q_ab(v, :) for a fixed variance v, all correlations c.

  Mirrors nngp._fill_qab_slice:
    log w(z1,z2) = -(z1^2 + z2^2 - 2 z1 z2 c) / (2 v (1 - c^2)),
  normalised over the (z1, z2) grid (log-sum-exp for stability), then
    q_ab = sum phi(z1) phi(z2) w(z1, z2).
  Processed in small correlation chunks so peak memory stays ~100 MB.
  """
  phi_name, var, chunk = args
  z, _, corr_ab = grid_axes()
  phi_z = NP_ACTIVATIONS[phi_name](z)
  sq = (z[:, None]**2 + z[None, :]**2)[None]        # [1, g, g]
  cross = (z[:, None] * z[None, :])[None]           # [1, g, g]
  pp = (phi_z[:, None] * phi_z[None, :])[None]      # [1, g, g]
  out = np.empty(corr_ab.shape[0])
  for s in range(0, corr_ab.shape[0], chunk):
    c = corr_ab[s:s + chunk, None, None]            # [k, 1, 1]
    logw = -(sq - 2.0 * c * cross) / (2.0 * var * (1.0 - c**2))
    logw -= logw.max(axis=(1, 2), keepdims=True)
    w = np.exp(logw)
    out[s:s + chunk] = (pp * w).sum(axis=(1, 2)) / w.sum(axis=(1, 2))
  return out


def compute_grid(phi_name, rows=None, processes=None, chunk=10):
  z, var_aa, corr_ab = grid_axes()
  idx = np.arange(len(var_aa)) if rows is None else np.asarray(rows)
  phi_z = NP_ACTIVATIONS[phi_name](z)
  qaa = np.array([qaa_row(phi_z, z, v) for v in var_aa[idx]])
  processes = processes or max(1, min(4, multiprocessing.cpu_count()))
  t0 = time.time()
  pool = multiprocessing.Pool(processes)
  qab = []
  for i, row in enumerate(pool.imap(qab_row,
                                    [(phi_name, v, chunk) for v in var_aa[idx]])):
    qab.append(row)
    if (i + 1) % 50 == 0 or i + 1 == len(idx):
      print('  %s: %d/%d variance rows (%.0fs)' % (phi_name, i + 1, len(idx),
                                                  time.time() - t0))
  pool.close()
  return var_aa, corr_ab, qaa, np.stack(qab), idx


def grid_filename(phi_name):
  # Same naming rule as NNGPKernel.get_grid (n_var == n_corr bump omitted
  # because 501 != 500).
  return 'grid_{}_ng{}_ns{}_nc{}_mv{}_mg{}'.format(
      phi_name, N_GAUSS, N_VAR, N_CORR, MAX_VAR, MAX_GAUSS)


def save_grid(phi_name, grid_dir):
  var_aa, corr_ab, qaa, qab, _ = compute_grid(phi_name)
  path = os.path.join(grid_dir, grid_filename(phi_name))
  # nngp.py does np.save of a 4-tuple -> 1-D object array; keep that format
  # so its np.load(..., allow_pickle=True) path works unchanged.
  obj = np.empty(4, dtype=object)
  obj[:] = [var_aa, corr_ab, qaa, qab]
  with open(path, 'wb') as f:
    np.save(f, obj, allow_pickle=True)
  print('Wrote', path)
  return path


def relu_closed_form(var, c):
  """Arc-cosine kernel (Cho & Saul 2009), the closed form for ReLU:
  E[relu(z1) relu(z2)] = v/(2 pi) * (sqrt(1 - c^2) + (pi - arccos c) c)."""
  return var / (2 * np.pi) * (np.sqrt(1 - c**2) + (np.pi - np.arccos(c)) * c)


def validate(grid_dir):
  rows = np.arange(0, N_VAR, 50)  # 11 variance rows spread over [1e-8, 100]
  ok = True
  for name in ('relu', 'tanh'):
    shipped = np.load(os.path.join(grid_dir, grid_filename(name)),
                      allow_pickle=True, encoding='latin1')
    _, _, qaa, qab, idx = compute_grid(name, rows=rows)
    d_qaa = np.max(np.abs(qaa - shipped[2][idx]))
    d_qab = np.max(np.abs(qab - shipped[3][idx]))
    print('[validate] %-4s  max|our - shipped|: q_aa %.2e   q_ab %.2e' %
          (name, d_qaa, d_qab))
    ok &= d_qab < 1e-6 and d_qaa < 1e-6
  # ReLU vs. exact arc-cosine kernel.  Checked for v in {1, 2, 5}: the
  # regime this project uses (unit-normalised inputs, sigma_w^2 = 2,
  # sigma_b^2 = 0.2 keep pre-activation variances around 2-3).
  # NOTE (finding): the repo integrates on a *fixed* z grid in [-10, 10]
  # regardless of v, so for large v the Gaussian tails are cut off.  Relative
  # error vs. the closed form is ~1e-4 at v=5, ~1e-2 at v=10 and ~9e-2 at
  # v=20.  Not an issue here, but it would matter for large sigma_w^2 / depth.
  z, var_aa, corr_ab = grid_axes()
  _, _, _, qab, idx = compute_grid('relu', rows=[5, 10, 25])
  exact = relu_closed_form(var_aa[idx][:, None], corr_ab[None, :])
  rel = np.max(np.abs(qab - exact) / var_aa[idx][:, None])
  print('[validate] relu vs closed-form arc-cosine kernel (v <= 5): '
        'max relative error %.2e' % rel)
  ok &= rel < 1e-3
  # GELU has no simple closed form, so spot-check the shipped GELU grid
  # against a Monte Carlo estimate of E[gelu(z1) gelu(z2)] (fixed seed).
  gpath = os.path.join(grid_dir, grid_filename('gelu'))
  if os.path.exists(gpath):
    g = np.load(gpath, allow_pickle=True, encoding='latin1')
    rng = np.random.RandomState(0)
    worst = 0.0
    for vi, ci in [(5, 250), (10, 400), (15, 490), (10, 50)]:
      v, c = g[0][vi], g[1][ci]
      e = rng.randn(400000, 2)
      z1 = np.sqrt(v) * e[:, 0]
      z2 = np.sqrt(v) * (c * e[:, 0] + np.sqrt(1 - c**2) * e[:, 1])
      prod = _np_gelu(z1) * _np_gelu(z2)
      zscore = abs(prod.mean() - g[3][vi, ci]) / (prod.std() / np.sqrt(len(prod)))
      print('[validate] gelu grid q_ab(v=%.1f, c=%+.3f) = %.5f   MC = %.5f'
            '   |z| = %.2f' % (v, c, g[3][vi, ci], prod.mean(), zscore))
      worst = max(worst, zscore)
    ok &= worst < 4.0
  print('[validate] %s' % ('PASSED' if ok else 'FAILED'))
  return ok


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  p.add_argument('--nonlinearity', default='gelu',
                 choices=sorted(NP_ACTIVATIONS))
  p.add_argument('--grid_dir', default='./grid_data')
  p.add_argument('--validate', action='store_true')
  a = p.parse_args()
  if a.validate:
    raise SystemExit(0 if validate(a.grid_dir) else 1)
  save_grid(a.nonlinearity, a.grid_dir)
