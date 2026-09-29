r"""Project 2 extension: does Figure 3's uncertainty/error relationship hold
for GELU, an activation the paper never tested?

Produces (in --output_dir):
  ext_cmap.png          the kernel's correlation map c_out = f(c_in) for one
                        layer, tanh vs ReLU vs GELU at the Figure 3
                        hyperparameters -- *why* the kernels differ
                        (ties to the paper's Sec. 3.2 signal-propagation view).
  ext_sweep.png         accuracy, binned correlation and calibration slope vs.
                        training-set size for tanh / ReLU / GELU.
  ext_metrics.csv       all numbers behind ext_sweep.png.

Hyperparameters are held at the paper's Figure 3 values (depth=3,
sigma_w^2=2.0, sigma_b^2=0.2) so the only thing that changes is phi.

Run from /nngp inside the container:
  python extension_gelu.py --train_sizes=500,1000,2000,5000 --num_eval=10000 \
      --data_dir=./data/mnist --output_dir=/nngp/output
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import activations
import make_grid
import uncertainty_plot as up  # defines the shared flags (num_eval, grids...)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_sizes', '500,1000,2000,5000',
                    'Comma-separated training-set sizes to sweep.')
flags.DEFINE_string('output_dir', '/nngp/output', 'Where to write outputs.')
flags.DEFINE_string('extra_metrics_csv', '',
                    'Optional metrics CSV from uncertainty_plot.py whose rows '
                    'are added to the sweep (avoids recomputing them).')
flags.DEFINE_string('ext_nonlinearities', 'tanh,relu,gelu',
                    'Activations compared in the extension.')


def _style():
  for s in ('seaborn-v0_8-darkgrid', 'seaborn-darkgrid'):
    if s in plt.style.available:
      plt.style.use(s)
      return


def correlation_map(name, weight_var, bias_var, q_in):
  """One layer of Eq. 4 in correlation form, straight from the lookup grid.

  With both inputs at pre-activation variance q_in and correlation c_in:
     q_out = sigma_b^2 + sigma_w^2 * q_aa(q_in)
     c_out = (sigma_b^2 + sigma_w^2 * q_ab(q_in, c_in)) / q_out
  This is the "C-map" of the signal-propagation papers the NNGP paper builds
  on in Sec. 3.2 (Poole et al. 2016; Schoenholz et al. 2017).
  """
  g = np.load(os.path.join(FLAGS.grid_path, make_grid.grid_filename(name)),
              allow_pickle=True, encoding='latin1')
  var_grid, corr_grid, qaa, qab = g
  i = np.argmin(np.abs(var_grid - q_in))  # nearest grid row (spacing 0.2)
  q_out = bias_var + weight_var * qaa[i]
  c_out = (bias_var + weight_var * qab[i]) / q_out
  return corr_grid, c_out, q_out


def plot_cmap(hparams, names, path):
  _style()
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
  # Layer-0 variance for unit-normalised inputs (nngp.py k_diag):
  # q0 = sigma_w^2 * 1 + sigma_b^2.
  q0 = hparams.weight_var + hparams.bias_var
  for n in names:
    _, label, color = activations.get_activation(n)
    c_in, c_out, q_out = correlation_map(n, hparams.weight_var,
                                         hparams.bias_var, q0)
    axes[0].plot(c_in, c_out, color=color, lw=2,
                 label='%s (q_out=%.2f)' % (label, q_out))
    # Iterate the map `depth` times from the actual input correlation range.
    c = np.linspace(-0.2, 0.99, 200)  # typical MNIST pairwise corr range
    cur, q = c.copy(), q0
    for _ in range(hparams.depth):
      grid_c, cm, q = correlation_map(n, hparams.weight_var,
                                      hparams.bias_var, q)
      cur = np.interp(cur, grid_c, cm)
    axes[1].plot(c, cur, color=color, lw=2, label=label)
  for ax in axes:
    ax.plot([-1, 1], [-1, 1], 'k--', lw=0.8, label='identity')
    ax.set_xlabel('input correlation $c_{in}$')
    ax.legend(loc='lower right', fontsize=9)
  axes[0].set_ylabel('$c_{out}$ after one layer')
  axes[0].set_title('One-layer correlation map ($\\sigma_w^2$=%.1f, '
                    '$\\sigma_b^2$=%.1f)' % (hparams.weight_var,
                                             hparams.bias_var))
  axes[1].set_ylabel('$c$ after %d layers' % hparams.depth)
  axes[1].set_xlabel('layer-0 kernel correlation $c^0$')
  axes[1].set_title('Kernel correlation at depth %d (the GP kernel)' %
                    hparams.depth)
  axes[1].set_xlim(-0.2, 1.0)
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)
  tf.logging.info('Saved %s', path)


def plot_sweep(rows, names, path):
  _style()
  fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
  keys = [('accuracy', 'Test accuracy'),
          ('corr_binned', 'Corr(variance, MSE), binned by 100'),
          ('calib_slope', 'Calibration slope MSE/variance (1 = calibrated)'),
          ('calib_slope_ml', 'Same, after ML kernel-amplitude rescale')]
  for n in names:
    _, label, color = activations.get_activation(n)
    rs = sorted([r for r in rows if r['nonlinearity'] == n],
                key=lambda r: r['num_train'])
    x = [r['num_train'] for r in rs]
    for ax, (k, title) in zip(axes, keys):
      ax.plot(x, [r[k] for r in rs], 'o-', color=color, label=label, lw=2)
      ax.set_title(title)
  sizes = sorted(set(r['num_train'] for r in rows))
  for ax in axes:
    ax.set_xscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.minorticks_off()
    ax.set_xlabel('training points (MNIST, 10k test points)')
    ax.legend()
  axes[1].set_ylim(0.9, 1.0)  # don't visually exaggerate small differences
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)
  tf.logging.info('Saved %s', path)


def main(argv):
  del argv
  hparams = up.set_default_hparams().parse(FLAGS.hparams or
                                           'depth=3,weight_var=2.0,bias_var=0.2')
  names = [s.strip() for s in FLAGS.ext_nonlinearities.split(',') if s.strip()]
  tf.gfile.MakeDirs(FLAGS.output_dir)

  plot_cmap(hparams, names, os.path.join(FLAGS.output_dir, 'ext_cmap.png'))

  rows = []
  for n_train in [int(s) for s in FLAGS.train_sizes.split(',')]:
    FLAGS.num_train = n_train
    data = up.load_data()
    for n in names:
      pred, act, acc, amp = up.compute_uncertainty_and_error(
          hparams, n, data[0], data[1], data[4], data[5])
      s = up.summarize(pred, act, FLAGS.bin_size, amp)
      s.update(nonlinearity=n, num_train=n_train, accuracy=acc)
      rows.append(s)
      print('EXT %-5s n_train=%-5d acc=%.4f corr_binned=%.4f corr_raw=%.4f '
            'calib_slope=%.3f calib_slope_ml=%.3f' % (
                n, n_train, acc, s['corr_binned'], s['corr_raw'],
                s['calib_slope'], s['calib_slope_ml']))

  # Optionally fold in rows already computed by uncertainty_plot.py (the
  # largest training size) instead of recomputing those GPs.
  if FLAGS.extra_metrics_csv and os.path.exists(FLAGS.extra_metrics_csv):
    with open(FLAGS.extra_metrics_csv) as f:
      for r in csv.DictReader(f):
        if r['nonlinearity'] in names and r['dataset'] == 'mnist':
          row = dict((k, float(v)) for k, v in r.items()
                     if k not in ('dataset', 'nonlinearity'))
          row.update(nonlinearity=r['nonlinearity'],
                     num_train=int(r['num_train']))
          rows.append(row)

  cols = ['nonlinearity', 'num_train', 'accuracy', 'corr_binned', 'corr_raw',
          'spearman_raw', 'calib_slope', 'ml_amplitude', 'calib_slope_ml',
          'mean_pred_var', 'mean_mse']
  with open(os.path.join(FLAGS.output_dir, 'ext_metrics.csv'), 'w') as f:
    w = csv.writer(f)
    w.writerow(cols)
    for r in rows:
      w.writerow([r[c] if isinstance(r[c], (str, int)) else '%.6g' % r[c]
                  for c in cols])
  plot_sweep(rows, names, os.path.join(FLAGS.output_dir, 'ext_sweep.png'))


if __name__ == '__main__':
  tf.app.run(main)
