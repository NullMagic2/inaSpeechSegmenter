#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2018 Ina (David Doukhan - http://www.ina.fr/)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
This module provides utility functions for preparing log-domain probabilities used in Viterbi decoding
within the VBx segmentation pipeline. It includes:

- pred2logemission: Converts binary prediction outputs into log emission probabilities, ensuring numerical
  stability via an epsilon value.
- log_trans_exp: Generates a 2x2 log-domain transition matrix with an exponential decay based on a given factor,
  allowing for custom diagonal costs.
- diag_trans_exp: Creates a square transition matrix of arbitrary dimension where off-diagonal entries are
  set to a constant negative cost (computed as -exp * log(10)) and diagonal entries are zero, promoting state
  continuity during decoding.

These functions are essential for setting up the emission and transition probability matrices required for
constrained Viterbi decoding.
"""

import numpy as np


def pred2logemission(pred, eps=1e-10):
    """
    Converts a binary prediction array into a log emission probability matrix.

    For each prediction value:
      - If the value is 0, the function sets the probability of the 0th class to (1 - eps) and the 1st class to eps.
      - If the value is 1, it sets the probability of the 1st class to (1 - eps) and the 0th class to eps.
    This results in a two-column probability matrix where each row sums to 1 (within numerical stability limits).
    Finally, the probabilities are transformed into the log domain using the natural logarithm.

    This function is typically used to prepare the emission probabilities for Viterbi decoding, ensuring that
    binary predictions (e.g., from a neural network) are in a log probability format that the decoding algorithm
    can process reliably.
    """
    pred = np.array(pred)
    ret = np.ones((len(pred), 2)) * eps
    ret[pred == 0, 0] = 1 - eps
    ret[pred == 1, 1] = 1 - eps
    return np.log(ret)

def log_trans_exp(exp,cost0=0, cost1=0):
    """
    Creates a 2x2 log-domain transition matrix for binary state transitions with an exponential decay cost.

    The function computes a base transition cost as -exp * log(10), which corresponds to a probability of 10**(-exp)
    in the linear domain. It fills a 2x2 matrix with this cost for off-diagonal transitions, while allowing the diagonal
    (self-transition) costs to be explicitly set to cost0 and cost1, respectively. This matrix is used during Viterbi
    decoding to penalize or favor certain state transitions based on the specified costs.

    Parameters:
        exp (float): The exponent factor to compute the base transition cost.
        cost0 (float): The log-domain cost for self-transition of state 0 (default is 0).
        cost1 (float): The log-domain cost for self-transition of state 1 (default is 0).

    Returns:
        numpy.ndarray: A 2x2 transition matrix in the log domain.
    """
    # transition cost is assumed to be 10**-exp
    cost = -exp * np.log(10)
    ret = np.ones((2,2)) * cost
    ret[0,0]= cost0
    ret[1,1]= cost1
    return ret

def diag_trans_exp(exp, dim):
    """
    Creates a square log-domain transition matrix of size dim x dim for Viterbi decoding.

    This function computes a constant off-diagonal transition cost as -exp * log(10) (representing the cost of switching states)
    and sets the diagonal elements (representing self-transitions) to zero. The resulting matrix is used in decoding to encourage
    state continuity by penalizing transitions between different states uniformly.
    """
    cost = -exp * np.log(10)
    ret = np.ones((dim, dim)) * cost
    for i in range(dim):
        ret[i, i] = 0
    return ret
