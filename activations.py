# Written for APPM 5750 Project 2 (Fall 2026) by Atharva Zodpe.
# Not part of the original brain-research/nngp release; this file is new.
# It builds on that repository's public API (Apache 2.0) and on the paper
# cited below.  Developed with LLM assistance -- see PROMPTS.md.
#
# Project 2 extension: activation functions for the NNGP kernel.
#
# The NNGP kernel recursion (Lee et al. 2018, Section 2.3, Eq. 4) is
#
#     K^l(x, x') = sigma_b^2 + sigma_w^2 * F_phi(K^{l-1}(x,x'), K^{l-1}(x,x),
#                                                 K^{l-1}(x',x'))
#
# where F_phi(.) = E[phi(z1) phi(z2)] for (z1, z2) jointly Gaussian with the
# layer-(l-1) covariance.  The paper only evaluates phi = ReLU and phi = tanh.
# Section 2.5 ("Efficient implementation") explains that for a *general* phi
# the expectation is computed numerically once, on a lookup grid over
# (variance, correlation), and then linearly interpolated.  nngp.py implements
# that general scheme (`_compute_qmap_grid`), so adding a new activation only
# needs (1) a TF function for phi and (2) its lookup grid.
#
# IMPORTANT implementation detail: nngp.NNGPKernel names the grid file on disk
# after `nonlin_fn.__name__` (e.g. grid_tanh_ng501_...).  A lambda would be
# called "<lambda>", so every activation here is a *named* def.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def gelu(x):
  """Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016).

  GELU(x) = x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2))),
  where Phi is the standard normal CDF.  This is the exact form (not the tanh
  approximation), used in BERT/GPT-style transformers.  The NNGP paper
  predates its widespread use and never tests it.

  Like ReLU it is unbounded above and ~0 for very negative inputs, but it is
  smooth and non-monotonic (dips to about -0.17 near x = -0.75).
  """
  return 0.5 * x * (1.0 + tf.erf(x / np.sqrt(2.0).astype(np.float64)))


# Map from the --nonlinearities flag value to (tf function, legend label,
# plot colour).  tanh/relu colours match the in-class figure; GELU is green.
ACTIVATIONS = {
    'tanh': (tf.tanh, 'Tanh', '#e8746c'),
    'relu': (tf.nn.relu, 'ReLU', '#3b5b92'),
    'gelu': (gelu, 'GELU', '#2e8b57'),
}


def get_activation(name):
  if name not in ACTIVATIONS:
    raise NotImplementedError(
        'Unknown nonlinearity %r; choose from %s' % (name, sorted(ACTIVATIONS)))
  return ACTIVATIONS[name]
