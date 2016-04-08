import numpy as np
import scipy.integrate
from sklearn.neighbors import KernelDensity

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
Module for implementing prior distributions on ensembles
"""


class BasePrior(object):
    """
    Abstract base class for prior distributions
    """

    def calc_prior_logp(self, *args):
        """
        Calculate the prior log probability for a list of Structures that make
        up an ensemble
        :param args: specific arguments
        :return: log probability of ensemble in this prior distribution
        """
        raise NotImplementedError

    def get_arg(self, struct):
        """
        Build the arguments required to calculate the prior given a
        single structure. So the input to calc_prior_logp shoudl be a
        list of these args
        :param struct: a Structure object
        :return:
        """
        raise NotImplementedError


class UniformPrior(BasePrior):
    """
    Uniform prior distribuiton across space of ensembles
    """

    def __init__(self, n):
        """
        :param n: number of candidate ensembles
        """
        self.n_ = n
        super(UniformPrior, self).__init__()

    def calc_prior_logp(self):
        """
        Probability is just 1/self.n_. See BasePrior for more info
        :return:
        """
        return np.log(1. / self.n_)

    def get_arg(self, struct=None):
        """
        No arguments for this prior. See BasePrior for more info
        :param struct: a Structure object
        :return: None
        """
        return None


class QuasiHarmonicPrior(BasePrior):
    """
    Calculate Kullback-Liebler divergence of an input ensemble to a reference
    ensemble based on the quasi-harmonic approximation, which assumes that
    all degrees of freedom can be represented as a high-dimensional normal
    distribution. The QH approximation was introduced in
    Macromolecules, 1981, 14 (2) pg. 325-332 and the Kullback-Liebler
    treatment of reference priors is taken from J. Chem. Phys. 143, 243150
    (2015).
    """

    def __init__(self, ref_array):
        """
        Requires an array of the values of the degrees of freedom of the
        system for each structure
        """
        self.refMu_ = np.mean(ref_array, axis=0)
        self.refCov_ = np.cov(ref_array, rowvar=0)
        super(QuasiHarmonicPrior, self).__init__()

    @staticmethod
    def normal_kl(test_mu, test_cov, ref_mu, ref_cov):
        """
        Calculate the Kullback-Liebler divergence between a test normal
        distribution and a reference normal distribution
        :param test_mu: mean of test distribution
        :param test_cov: covariance matrix of test distribution
        :param ref_mu: mean of reference distribution
        :param ref_cov: covariance matrix of reference distribution
        :return: KL divergence between test and reference distributions
        """
        k = len(ref_mu)
        det1 = np.linalg.det(ref_cov)
        det0 = np.linalg.det(test_cov)
        sig1_inv = np.linalg.inv(ref_cov)
        kl = np.trace(np.dot(sig1_inv, test_cov))
        mu_diff = ref_mu - test_mu
        a = np.dot(np.transpose(mu_diff), sig1_inv)
        b = np.dot(a, mu_diff)
        kl += b
        kl -= k
        c = np.log(det1 / det0)
        kl += c
        kl *= 0.5
        return kl

    def calc_prior_logp(self, test_array):
        """
        Defined as the negative KL divergence between the test and
        reference ensemble. See BasePrior for more info.
        :param test_array: list of degrees of freedom values for test ensemble
        """
        test_array = np.array(test_array)
        test_mu = np.mean(test_array, axis=0)
        test_cov = np.cov(test_array, rowvar=0)

        kl = QuasiHarmonicPrior.normal_kl(test_mu, test_cov,
                                          self.refMu_, self.refCov_)
        log_prior = kl
        return log_prior

    def get_arg(self, struct):
        """
        Assumes only degrees of freedom are dihedral angles.
        See BasePrior for more info
        :param struct: a Structure object
        :return: vector of dihedral angles
        """
        all_dihed = struct.dihed_
        col = 0
        dihed_row = np.zeros(len(all_dihed))
        for j in range(1, max(all_dihed.keys())):
            if j == 1:
                dihed_row[col] = all_dihed[j][1]
                col += 1
            elif j == max(all_dihed.keys()) - 1:
                dihed_row[col] = all_dihed[j][0]
                col += 1
            else:
                dihed_row[col] = all_dihed[j][0]
                col += 1
                dihed_row[col] = all_dihed[j][1]
                col += 1
        return dihed_row


class EnergyPrior(BasePrior):
    """
    Calculate Kullback-Liebler divergence of an input ensemble to a reference
    ensemble based a Kernel Density Approximation of the energy distribution
    """

    def __init__(self, ref_energies, pdb_dict=None):
        """
        Constructor requires energies of reference ensemble. Can optionally
        including a {pdbname: energy} dict that can be used to find
        the energy of any input structure
        :param ref_energies: list of energies of reference ensemble
        :param pdb_dict: optional {pdbname: energy} dict
        """
        vals = np.array(ref_energies)
        vals -= np.mean(vals)
        vals /= np.std(vals)
        vals = vals.reshape((vals.shape[0], 1))

        n = len(vals)
        d = 1

        bdwith = n ** (-1. / (d + 4))  # Scott's rule
        self.refKDE_ = KernelDensity(bandwidth=bdwith)
        self.refKDE_.fit(vals)

        # integration range:
        self.intRange_ = [np.min(vals) - 10 * bdwith,
                          np.max(vals) + 10 * bdwith]

        self.pdbDict_ = pdb_dict

        super(EnergyPrior, self).__init__()

    @staticmethod
    def kde_kl(kde1, kde2, int_range):
        """
        Find Kullback-Liebler divergence between two 1 dimensional
        Kernel Density Approxmiations over a give range
        :param kde1: a 1d KernelDensity (p(x))
        :param kde2: a 1d KernelDensity (q(x))
        :param int_range: min and max of integration
        :return: KL divergence
        """

        def func(z):
            """
            Function integrate. Implements inside of KL divergence equation
            :param z: a value for which probabilities exist
            :return: result of KL for this value
            """
            z = np.array(z)
            px = np.exp(kde1.score_samples(z))
            qx = np.exp(kde2.score_samples(z))
            kl = px * np.log(px / qx)
            return kl

        kldiv = scipy.integrate.quad(func, int_range[0], int_range[1])[0]
        return kldiv

    def calc_prior_logp(self, test_energies):
        """
        Defined as the negative KL divergence between the test and
        reference Kernel Density Approximations. See BasePrior for more info.
        :param test_energies: list of test ensemble energies
        """
        test_vals = np.array(test_energies)
        test_vals -= np.mean(test_vals)
        test_vals /= np.std(test_vals)
        test_vals = test_vals.reshape((test_vals.shape[0], 1))
        n = len(test_vals)
        d = 1
        bdwith = n ** (-1. / (d + 4))  # Scott's rule
        test_kde = KernelDensity(bandwidth=bdwith)
        test_kde.fit(test_vals)
        logp = EnergyPrior.kde_kl(test_kde, self.refKDE_, self.intRange_)
        return -logp

    def get_arg(self, struct):
        """
        Returns the energy of the structure, either from the energy stored
        in the Structure object or from self.pdbDict_. See BasePrior for more
        info
        :param struct: a Structure object
        :return: the energy of the structure
        """
        if struct.energy_ is None and self.pdbDict_ is not None:
            return self.pdbDict_[struct.pdb_]
        else:
            return struct.energy_
