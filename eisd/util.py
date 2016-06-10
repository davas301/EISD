from scipy.stats import norm
import numpy as np
import time
import matplotlib.pyplot as plt

"""
Copyright (c) 2016, Teresa Head-Gordon and David Brookes
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of UC Berkeley nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Teresa Head-Gordon BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__author__ = 'David Brookes'
__date__ = '4/4/16'

"""
Module containing various useful methods and variables
"""

# One letter codes for the 20 amino acids, sorted alphabetically
RESIDUE_1_CODES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# three letter codes for the 20 amino acids (same order as one letter codes)
RESIDUE_3_CODES = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE',
                   'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
                   'MET', 'ASN', 'PRO', 'GLN', 'ARG',
                   'SER', 'THR', 'VAL', 'TRP', 'TYR']

# RMSD of true shift values vs SHIFTX2 predictions for every atom type
# listed on their website.
SHIFTX2_RMSD = {
    'N': 1.2328, 'CA': 0.3836, 'CB': 0.5329, 'C': 0.5096, 'H': 0.2351,
    'HA': 0.1081, 'CD': 0.5059, 'CD1': 1.0113, 'CD2': 1.1812, 'CE': 0.3690,
    'CE1': 0.9168, 'CE2': 0.8092, 'CG': 0.6635, 'CG1': 0.7835, 'CG2': 0.9182,
    'CZ': 1.1673, 'HA2': 0.2732, 'HA3': 0.3007, 'HB': 0.1464, 'HB2': 0.1814,
    'HB3': 0.2109, 'HD1': 0.1760, 'HD2': 0.1816, 'HD3': 0.1796, 'HE': 0.3342,
    'HE1': 0.1949, 'HE2': 0.1590, 'HE3': 0.3219, 'HG': 0.3826, 'HG1': 0.1489,
    'HG12': 0.1881, 'HG13': 0.1269, 'HG2': 0.1401, 'HG3': 0.1689, 'HZ': 0.2610
}


def normal_loglike(x, mu, sig):
    """
    Returns the log likelihood of a value x in a normal distribution with
    mean mu and stdev sig.

    :param x: list of values to calculate probability of
    :param mu: mean of distribuiton
    :param sig: standard deviation of distribution
    """
    return norm.logpdf(x, loc=mu, scale=sig)


def normal_log_deriv(x, mu, sig):
    """
    Derivative of the log of the normal distribution with respect to x
    :param x: value to be tested
    :param mu: mean
    :param sig: variance
    :return: d/dx
    """
    return (-2 * x + mu) / (2 * sig ** 2)


def timeit(f):
    """
    Timing decorator for a function f
    :param f: a function
    :return: timing
    """

    def timed(*args, **kw):
        """
        A timed function
        :param args:
        :param kw:
        :return:
        """
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print 'func:%r took: %2.4f sec' % \
              (f.__name__, te - ts)
        return result

    return timed



