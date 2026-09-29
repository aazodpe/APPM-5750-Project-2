# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for APPM 5750 Project 2 (Fall 2026): started from the in-class
# uncertainty_plot.py; added paper cross-references, GELU support (see
# activations.py / make_grid.py), accuracy + calibration metrics, a CSV
# metrics dump, zero-mean one-hot CIFAR labels, and a --data_dir for the
# vendored MNIST files.
r"""Reproduce Figure 3 of "Deep Neural Networks as Gaussian Processes"
(Lee et al., ICLR 2018, https://arxiv.org/abs/1711.00165).

Paper, Figure 3 caption: "The Bayesian nature of NNGP allows it to assign a
prediction uncertainty to each test point. This prediction uncertainty is
highly correlated with the empirical error on test points. The x-axis shows
the predicted MSE for test points, while the y-axis shows the realized MSE.
[...] each plotted point is an average over 100 test points, binned by
predicted MSE. The hyperparameters for the NNGP are depth=3, sigma_w^2=2.0,
and sigma_b^2=0.2."

Pipeline (each step is labelled in the code with the paper section):
  1. Load data; targets are one-hot with the mean subtracted (Sec. 3.1).
  2. Build the NNGP kernel K^L by the layer recursion (Sec. 2.3, Eq. 4-5),
     evaluated with the lookup-grid trick (Sec. 2.5)           -> nngp.py
  3. Exact GP regression: posterior mean and variance (Sec. 2.4, Eq. 8-9)
                                                                -> gpr.py
  4. Per test point: predicted MSE = posterior variance (Eq. 9 diagonal),
     realised MSE = mean over the 10 outputs of (mean - target)^2.
  5. Sort by predicted MSE, average in bins of 100, scatter (Fig. 3).

Examples (from /nngp inside the container):

  # Paper's two-colour figure (tanh + ReLU) on MNIST:
  python uncertainty_plot.py --num_train=5000 --num_eval=10000 \
      --hparams='depth=3,weight_var=2.0,bias_var=0.2' \
      --nonlinearities='tanh,relu' --data_dir=./data/mnist \
      --output_file=/nngp/output/fig3_mnist.png

  # Extension: add GELU (grid_data/grid_gelu_* built by make_grid.py):
  python uncertainty_plot.py ... --nonlinearities='tanh,relu,gelu'

  # CIFAR-10 (downloads ~170 MB on first use):
  python uncertainty_plot.py --dataset=cifar10 ...
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os.path

import matplotlib
matplotlib.use('Agg')  # no display available inside the container
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import activations
import gpr
import load_dataset
import nngp

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('hparams', '',
                    'Comma separated list of name=value hyperparameter '
                    'pairs to override the default setting. nonlinearity '
                    'here is used as-is unless --nonlinearities is set.')
flags.DEFINE_string('nonlinearities', '',
                    'Optional comma-separated list of nonlinearities to '
                    'run and overlay, e.g. "tanh,relu,gelu". Overrides the '
                    'nonlinearity in --hparams; all other hparams (depth, '
                    'weight_var, bias_var) are shared across runs, '
                    'matching how the paper produces Figure 3.')
flags.DEFINE_string('experiment_dir', '/tmp/nngp',
                    'Directory to put the experiment results.')
flags.DEFINE_string('grid_path', './grid_data',
                    'Directory to put or find the kernel lookup grids.')
flags.DEFINE_integer('num_train', 1000, 'Number of training data.')
flags.DEFINE_integer('num_eval', 1000,
                     'Number of test points to plot uncertainty for.')
flags.DEFINE_integer('bin_size', 100,
                     'Number of test points averaged into each plotted '
                     'point, binned by predicted variance (100 in the '
                     'paper).')
flags.DEFINE_integer('seed', 1234, 'Random number seed for data shuffling')
flags.DEFINE_string('dataset', 'mnist',
                    'Which dataset to use ["mnist", "cifar10"]')
flags.DEFINE_boolean('use_fixed_point_norm', False,
                     'Normalize input variance to fixed point variance')
flags.DEFINE_integer('n_gauss', 501,
                     'Number of gaussian integration grid. Choose odd '
                     'integer.')
flags.DEFINE_integer('n_var', 501, 'Number of variance grid points.')
flags.DEFINE_integer('n_corr', 500, 'Number of correlation grid points.')
flags.DEFINE_integer('max_var', 100, 'Max value for variance grid.')
flags.DEFINE_integer('max_gauss', 10, 'Range for gaussian integration.')
flags.DEFINE_string('output_file', '/nngp/output/uncertainty_fig3.png',
                    'Where to save the resulting plot.')
flags.DEFINE_string('paper_output_file', '',
                    'Optional second plot containing only tanh/ReLU (the '
                    'paper\'s two curves), made from the same runs.')
flags.DEFINE_string('metrics_file', '',
                    'Optional CSV to append per-nonlinearity metrics to.')

_DATASET_LABELS = {'mnist': 'MNIST', 'cifar10': 'CIFAR'}


def load_cifar10(num_train, mean_subtraction=True, num_valid=5000):
  """Loads CIFAR-10 as flattened, one-hot numpy arrays.

  Not part of the original repo's load_dataset.py (which only implements
  MNIST). Mirrors load_dataset.load_mnist's output: [N, 3072] float inputs,
  [N, 10] labels, train/valid/test splits.
  """
  # Bundled with TF 1.15's Keras; downloads to ~/.keras/datasets on first use.
  from tensorflow.keras.datasets import cifar10  # pylint: disable=g-import-not-at-top

  (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

  def _flatten(x):
    return x.reshape(x.shape[0], -1).astype(np.float64) / 255.0

  def _one_hot(y, num_classes=10):
    y = y.reshape(-1)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out

  x_train_full = _flatten(x_train_full)
  x_test = _flatten(x_test)
  y_train_full = _one_hot(y_train_full)
  y_test = _one_hot(y_test)

  if num_train + num_valid > x_train_full.shape[0]:
    raise ValueError(
        'num_train (%d) + validation holdout (%d) exceeds the CIFAR-10 '
        'training set size (%d).' % (
            num_train, num_valid, x_train_full.shape[0]))

  train_image = x_train_full[:num_train]
  train_label = y_train_full[:num_train]
  valid_image = x_train_full[-num_valid:]
  valid_label = y_train_full[-num_valid:]

  if mean_subtraction:
    # Sec. 3.1: "one-hot, zero-mean" regression targets (0.9 / -0.1), the
    # same preprocessing load_dataset.load_mnist applies to MNIST.  (The
    # in-class version only centred the images, not the labels.)
    mean = train_image.mean()
    label_mean = train_label.mean()
    train_image = train_image - mean
    valid_image = valid_image - mean
    x_test = x_test - mean
    train_label = train_label - label_mean
    valid_label = valid_label - label_mean
    y_test = y_test - label_mean

  return train_image, train_label, valid_image, valid_label, x_test, y_test


def load_data():
  """Step 1: data + zero-mean one-hot targets (paper Sec. 3.1)."""
  if FLAGS.dataset == 'mnist':
    # load_mnist subtracts the scalar mean from images *and* labels, giving
    # the paper's 0.9 / -0.1 targets.  Class-balanced subset of size
    # num_train; test set is the standard 10k MNIST test images.
    return load_dataset.load_mnist(
        num_train=FLAGS.num_train, use_float64=True, mean_subtraction=True,
        random_roated_labels=False)
  elif FLAGS.dataset == 'cifar10':
    return load_cifar10(num_train=FLAGS.num_train, mean_subtraction=True)
  raise NotImplementedError(FLAGS.dataset)


def set_default_hparams():
  return tf.contrib.training.HParams(
      nonlinearity='tanh', weight_var=1.3, bias_var=0.2, depth=2)


def _size_label(n):
  if n >= 1000 and n % 1000 == 0:
    return '%dk' % (n // 1000)
  return str(n)


def bin_by_predicted_mse(predicted_mse, actual_mse, bin_size):
  """Step 5: sort by predicted MSE and average every `bin_size` points.

  Figure 3 caption: "each plotted point is an average over 100 test points,
  binned by predicted MSE."  A single squared error is a very noisy draw;
  averaging 100 of them estimates the *expected* error at that uncertainty
  level, which is what the GP variance is a prediction of.
  """
  order = np.argsort(predicted_mse)
  pred_sorted = predicted_mse[order]
  act_sorted = actual_mse[order]

  n_bins = len(order) // bin_size
  if n_bins == 0:
    raise ValueError(
        'Not enough test points (%d) for bin_size=%d; lower --bin_size or '
        'raise --num_eval.' % (len(order), bin_size))

  pred_binned = pred_sorted[:n_bins * bin_size].reshape(n_bins, bin_size)
  act_binned = act_sorted[:n_bins * bin_size].reshape(n_bins, bin_size)
  return pred_binned.mean(axis=1), act_binned.mean(axis=1)


def compute_uncertainty_and_error(hparams, nonlinearity, train_image,
                                  train_label, test_image, test_label):
  """Steps 2-4 for one nonlinearity.

  Returns (predicted_mse, actual_mse, accuracy), the first two per test point.
  """
  nonlin_fn = activations.get_activation(nonlinearity)[0]

  # Fresh graph per nonlinearity so repeated runs in one process don't
  # collide on tensor/placeholder names.
  graph = tf.Graph()
  with graph.as_default():
    with tf.Session() as sess:
      # Step 2 -- Sec. 2.3, Eq. 4-5: K^0 = sigma_b^2 + sigma_w^2 x.x'/d_in,
      # then `depth` applications of
      #   K^l = sigma_b^2 + sigma_w^2 F_phi(K^{l-1}),
      # where F_phi is read off the precomputed (variance, correlation)
      # lookup grid by bilinear interpolation (Sec. 2.5).  The grid file is
      # grid_data/grid_<phi.__name__>_ng501_ns501_nc500_mv100_mg10.
      nngp_kernel = nngp.NNGPKernel(
          depth=hparams.depth,
          weight_var=hparams.weight_var,
          bias_var=hparams.bias_var,
          nonlin_fn=nonlin_fn,
          grid_path=FLAGS.grid_path,
          n_gauss=FLAGS.n_gauss,
          n_var=FLAGS.n_var,
          n_corr=FLAGS.n_corr,
          max_gauss=FLAGS.max_gauss,
          max_var=FLAGS.max_var,
          use_fixed_point_norm=FLAGS.use_fixed_point_norm)

      # Step 3 -- Sec. 2.4, Eq. 8-9: exact GP posterior
      #   mean  = K_{*,D} (K_{D,D} + s^2 I)^{-1} t                   (Eq. 8)
      #   var   = K_{*,*} - K_{*,D} (K_{D,D} + s^2 I)^{-1} K_{*,D}^T (Eq. 9)
      # gpr.py solves this with a Cholesky factorisation; s^2 is the tiny
      # "stability_eps" jitter (starts at 1e-10, raised only if Cholesky
      # fails), so the predictions are essentially noise-free.
      model = gpr.GaussianProcessRegression(
          train_image, train_label, kern=nngp_kernel)

      n_eval = min(FLAGS.num_eval, test_image.shape[0])
      tf.logging.info('[%s] Computing predictive mean/variance for %d test '
                      'points', nonlinearity, n_eval)
      mean_pred, var_pred, _ = model.predict(
          test_image[:n_eval], sess, get_var=True)

  # Step 4.  The 10 one-hot outputs are independent GPs sharing the same
  # kernel, so the Eq. 9 variance is identical across the 10 columns --
  # averaging over columns just returns that per-point variance.
  targets = test_label[:n_eval]
  actual_mse = np.mean((mean_pred - targets)**2, axis=1)
  predicted_mse = np.mean(var_pred, axis=1)
  # Classification = argmax of the regression output (Sec. 3.1).
  accuracy = np.mean(np.argmax(mean_pred, 1) == np.argmax(targets, 1))
  # (Extension diagnostic, not in the paper.)  The kernel's overall scale is
  # fixed by sigma_w^2 / sigma_b^2, not fitted to the labels, so the Eq. 9
  # variance can be off by a constant factor.  The maximum-likelihood
  # amplitude for K -> a*K is  a = t^T K^{-1} t / (n * n_outputs); gpr.py
  # already stores v = L^{-1} t (K = L L^T), so t^T K^{-1} t = ||v||^2.
  amplitude = float(np.sum(model.v_np**2) / model.v_np.size)
  return predicted_mse, actual_mse, accuracy, amplitude


def summarize(predicted_mse, actual_mse, bin_size, amplitude=None):
  """Numbers reported alongside the figure.

  corr_binned   Pearson r of the plotted (binned) points: the number the paper
                prints in its legend.
  corr_raw      Pearson r of the un-binned per-point values (much lower,
                because one squared error is a noisy draw).
  spearman_raw  rank correlation, per point.
  calib_slope   least-squares slope (through the origin) of binned MSE on
                binned variance.  1.0 = perfectly calibrated in scale; the
                paper's figure only claims correlation, not calibration.
  calib_slope_ml  same, after rescaling the variance by the max-likelihood
                kernel amplitude (see compute_uncertainty_and_error).
  """
  pb, ab = bin_by_predicted_mse(predicted_mse, actual_mse, bin_size)
  rank = lambda v: np.argsort(np.argsort(v))
  return {
      'corr_binned': np.corrcoef(pb, ab)[0, 1],
      'corr_raw': np.corrcoef(predicted_mse, actual_mse)[0, 1],
      'spearman_raw': np.corrcoef(rank(predicted_mse), rank(actual_mse))[0, 1],
      'calib_slope': float(np.dot(pb, ab) / np.dot(pb, pb)),
      'ml_amplitude': float('nan') if amplitude is None else amplitude,
      'calib_slope_ml': (float('nan') if amplitude is None else
                         float(np.dot(pb, ab) / np.dot(pb, pb) / amplitude)),
      'mean_pred_var': float(np.mean(predicted_mse)),
      'mean_mse': float(np.mean(actual_mse)),
  }


def make_figure3(runs, output_file, title):
  """Step 5: paper-styled scatter of binned variance vs. binned MSE.

  `runs` maps nonlinearity -> (predicted_mse, actual_mse, accuracy, amp), raw
  per-point arrays; binning and the correlation are computed here so the
  legend matches what is plotted (legend format copies the paper's
  "Tanh-corr:0.xxxx").
  """
  if os.path.dirname(output_file):
    tf.gfile.MakeDirs(os.path.dirname(output_file))

  # Style name changed between matplotlib versions; fall back gracefully.
  for style_name in ('seaborn-v0_8-darkgrid', 'seaborn-darkgrid'):
    if style_name in plt.style.available:
      plt.style.use(style_name)
      break
  fig, ax = plt.subplots(figsize=(7, 6))

  for nonlinearity, (predicted_mse, actual_mse, _, _) in runs.items():
    _, label, color = activations.get_activation(nonlinearity)
    pred_binned, act_binned = bin_by_predicted_mse(
        predicted_mse, actual_mse, FLAGS.bin_size)
    corr = np.corrcoef(pred_binned, act_binned)[0, 1]
    ax.scatter(
        pred_binned, act_binned,
        s=28, alpha=0.8, color=color,
        edgecolors='white', linewidths=0.4,
        label='%s-corr:%.4f' % (label, corr))

  ax.set_xlabel('Output variance')
  ax.set_ylabel('MSE')
  ax.set_title(title)
  ax.legend(loc='upper left', frameon=True)
  fig.tight_layout()

  with tf.gfile.Open(output_file, 'wb') as f:
    fig.savefig(f, format='png', dpi=150)
  plt.close(fig)
  tf.logging.info('Saved plot to %s', output_file)


def write_metrics(runs, hparams):
  if not FLAGS.metrics_file:
    return
  new = not tf.gfile.Exists(FLAGS.metrics_file)
  with open(FLAGS.metrics_file, 'a') as f:
    w = csv.writer(f)
    cols = ['dataset', 'num_train', 'num_eval', 'nonlinearity', 'depth',
            'weight_var', 'bias_var', 'accuracy', 'corr_binned', 'corr_raw',
            'spearman_raw', 'calib_slope', 'ml_amplitude', 'calib_slope_ml',
            'mean_pred_var', 'mean_mse']
    if new:
      w.writerow(cols)
    for nl, (p, a, acc, amp) in runs.items():
      s = summarize(p, a, FLAGS.bin_size, amp)
      w.writerow([FLAGS.dataset, FLAGS.num_train, len(p), nl, hparams.depth,
                  hparams.weight_var, hparams.bias_var, '%.4f' % acc] +
                 ['%.6g' % s[k] for k in cols[8:]])


def run(hparams):
  tf.gfile.MakeDirs(FLAGS.experiment_dir)
  tf.logging.info('Loading data')
  (train_image, train_label, _, _, test_image, test_label) = load_data()

  nonlinearities = ([s.strip() for s in FLAGS.nonlinearities.split(',') if
                     s.strip()] or [hparams.nonlinearity])

  runs = {}
  for nonlinearity in nonlinearities:
    runs[nonlinearity] = compute_uncertainty_and_error(
        hparams, nonlinearity, train_image, train_label, test_image,
        test_label)
    p, a, acc, amp = runs[nonlinearity]
    s = summarize(p, a, FLAGS.bin_size, amp)
    print('RESULT %-5s n_train=%-6d acc=%.4f corr(binned)=%.4f '
          'corr(raw)=%.4f calib_slope=%.3f calib_slope_ml=%.3f' % (
              nonlinearity, FLAGS.num_train, acc, s['corr_binned'],
              s['corr_raw'], s['calib_slope'], s['calib_slope_ml']))

  dataset_label = _DATASET_LABELS.get(FLAGS.dataset, FLAGS.dataset.upper())
  title = '%s %s-%s' % (dataset_label, '/'.join(
      activations.get_activation(n)[1] for n in nonlinearities),
                        _size_label(FLAGS.num_train))
  if FLAGS.output_file:
    make_figure3(runs, FLAGS.output_file, title)
  # Paper-faithful panel (tanh + ReLU only) from the same runs, so the
  # extension overlay and the reproduction don't recompute the same GPs.
  paper = [n for n in ('tanh', 'relu') if n in runs]
  if FLAGS.paper_output_file and paper:
    make_figure3(dict((n, runs[n]) for n in paper), FLAGS.paper_output_file,
                 '%s Tanh/ReLU-%s' % (dataset_label,
                                      _size_label(FLAGS.num_train)))
  write_metrics(runs, hparams)
  return runs


def main(argv):
  del argv  # Unused
  hparams = set_default_hparams().parse(FLAGS.hparams)
  run(hparams)


if __name__ == '__main__':
  tf.app.run(main)
