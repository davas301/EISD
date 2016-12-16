import numpy as np
from readutil import Measurement
from util import SHIFTX2_RMSD, normal_loglike, solve_3_eqs

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
Module for implementing back-calculators
"""


class BaseBackCalculator(object):
    """
    Abstract base class for back calculators

    :param nparams: number of nuisance parameters
    :param dtype: name of the data this back-calculator concerns (e.g. "shift" or "jcoup)
    """

    def __init__(self, nparams, dtype):
        self.dtype_ = dtype
        self.nParams_ = nparams

    def get_err_sig(self, data_id):
        """
        Get error standard deviation for a data point

        :param data_id: DataID object for the data point
        """
        raise NotImplementedError

    def get_default_params(self):
        """
        Get the default parameters for this back-calculator

        :return: List of default values (one for each param)
        """
        raise NotImplementedError

    def get_random_params(self):
        """
        Get values drawn randomly  from the distributions of this
        back-calculator's parameters

        :return: List of random values (one for each param)
        """
        raise NotImplementedError

    def get_random_err(self, data_id):
        """
        Get a value drawn from the error distribution of this back-calculator

        :param data_id: a DataID object
        :return: a random error taken from the distribution corresponding to
                 the input data id
        """
        raise NotImplementedError

    def get_default_err(self, data_id):
        """
        Get the default (i.e. most probable) error for this back-calculator

        :param data_id: a DataID object
        :return: the default error corresponding to the input data id
        """
        raise NotImplementedError

    def logp_params(self, params):
        """
        Return the log probability of a set of parameter values

        :param params: list of input parameters for this back-calculator
        :return: log probability of the input parameters
        """
        raise NotImplementedError

    def logp_err(self, err, data_id):
        """
        Return the log probability of an error

        :param err: the error value
        :param data_id: the DataID corresponding to the input error
        :return: the log probability of the input error
        """
        raise NotImplementedError

    def back_calc(self, xi, params):
        """
        Perform the back-calculation given a structural measurement and
        necessary parameters.

        :param xi: a Measurement returned by self.get_struct_data()
        :param params: list of nuisance parameter values for this back-calc
        :return: a Measurement object containing the back-calculated value
        """
        raise NotImplementedError

    def get_default_struct_val(self):
        """
        Returns a default structural value for the relevant
        structural information needed by this back-calculator.
        Used to initialize models

        :return: default Measurement object
        """
        raise NotImplementedError

    def calc_opt_params(self, n, precalc_params, sig_eps, exp_val):
        """
        Given all structural measurements, calculate the optimal
        paramater set and return the total probability with those
        parameters (i.e. the optimal function value). This version is a
        fast implementation that uses pre-computed values of structural
        measurements
        :param n: number of structures
        :param precalc_params: pre-calculated sums over structural measurements
        :param sig_eps: std of experimental error
        :param exp_val: experimental value
        :return: list of optimal parameters, optimal function value
        """
        raise NotImplementedError


class JCoupBackCalc(BaseBackCalculator):
    """
    Back calculator for J-coupling constants from dihedral angles
    Calculation is done with the Karplus equation:
    \$\$ J(\phi) = A\cos^2(\phi) + Bcos(\phi) + C \$\$
    Constructor requires parameters for the Gaussian random variables
    that will represent the coefficients in the Karplus equation
    and a dictionary containing the errors mean and standard deviations
    for every type of structural measurement. Default Karplus coeff means,
    stdevs and and error stdev are the values based on those in:
        Vuister and Bax, "Quantatative J Correlation: a new approach for
        meausuring homonuclear
        three bond J coupling constants in N15 eriched proteins"
        *J.Am.Chem.Soc*, **1993**, *115* (17). pp 7772:7777
    """

    def __init__(self, err_mu=0, err_sig=0.73, mu_a=6.51,
                 mu_b=-1.76, mu_c=1.60, sig_a=np.sqrt(0.14),
                 sig_b=np.sqrt(0.03), sig_c=np.sqrt(0.08)
                 ):
        self.muParams_ = [mu_a, mu_b, mu_c]
        self.sigParams_ = [sig_a, sig_b, sig_c]
        self.errMu_ = err_mu
        self.errSig_ = err_sig

        super(JCoupBackCalc, self).__init__(nparams=3, dtype="jcoup")

    def get_err_sig(self, data_id=None):
        """
        See BaseBackCalculator for more info

        :param data_id:
        :return:
        """
        return self.errSig_

    def get_random_params(self):
        """
        See BaseBackCalculator for more info

        :return:
        """
        params = [0, 0, 0]
        for i in range(len(params)):
            params[i] = np.random.normal(loc=self.muParams_[i],
                                         scale=self.sigParams_[i])
        return params

    def logp_params(self, params):
        """
        Parameters are represented as normals. See BaseBackCalculator for more
        info

        :param params:
        :return:
        """
        logp = 0
        for i in range(self.nParams_):
            logp += normal_loglike(params[i], mu=self.muParams_[i],
                                   sig=self.sigParams_[i])
        return logp

    def get_random_err(self, data_id=None):
        """
        data_id is optional for this back-calculator. See BaseBackCalculator for
        more info

        :param data_id:
        """
        return np.random.normal(loc=0, scale=self.errSig_)

    def get_default_params(self):
        """
        Return mean of param values. See BaseBackCalculator for more info

        :return:
        """
        return self.muParams_

    def get_default_err(self, data_id=None):
        """
        Default if mean of error distribution See BaseBackCalculator for more
        info

        :param data_id:
        :return:
        """
        return self.errMu_

    def logp_err(self, err, data_id=None):
        """
        Error is represemted as a normal. See BaseBackCalculator for more info

        :param err:
        :param data_id:
        """
        logp = normal_loglike(err, mu=self.errMu_, sig=self.errSig_)
        return logp

    def back_calc(self, xi, back_params):
        """
        Implementation of the Karplus equation. See BaseBackCalculator for more
        info

        :param xi:
        :param back_params:
        """
        phi = xi.val_[0]
        a = back_params[0]
        b = back_params[1]
        c = back_params[2]
        j = a * (np.cos(phi - (np.pi / 3))) ** 2
        j += b * np.cos(phi - (np.pi / 3)) + c
        return j

    def get_default_struct_val(self):
        """
        Default is [pi, pi] dihedral angle. See BaseBackCalculator for more info
        :return:
        """
        meas = Measurement(data_id=None,
                           val=np.array([np.pi, np.pi]))
        return meas

    def calc_opt_params(self, n, alphas, sig_eps, exp_val):
        """
        Analytically calculate the optimal A, B and C in the Karplus equation.
        For more info see BaseBackCalculator
        :param n: number of structures
        :param alphas: pre-calculated sum over cosines of phi angles
        :param sig_eps:
        :param exp_val:
        :return: optimal A, B, C in a list
        """
        alpha1 = alphas[0]
        alpha2 = alphas[1]

        a = np.zeros((3, 3))
        b = np.zeros((3,))

        b[0] = ((self.muParams_[0] / self.sigParams_[0] ** 2) + (
            (exp_val * alpha2) / (n * sig_eps ** 2)))

        b[1] = ((self.muParams_[1] / self.sigParams_[1] ** 2) + (
            (exp_val * alpha1) / (n * sig_eps ** 2)))

        b[2] = ((self.muParams_[2] / self.sigParams_[2] ** 2) + (
            exp_val / (sig_eps ** 2)))

        a[0, 0] = (
            (1 / self.sigParams_[0] ** 2) + (
                alpha2 ** 2 / (sig_eps ** 2 * n ** 2)))
        a[1, 1] = (
            (1 / self.sigParams_[1] ** 2) + (
                alpha1 ** 2 / (sig_eps ** 2 * n ** 2)))
        a[2, 2] = ((1 / self.sigParams_[2] ** 2) + (1 / (sig_eps ** 2)))

        a[0, 1] = (1 / (sig_eps ** 2 * n ** 2)) * alpha2 * alpha1
        a[1, 0] = (1 / (sig_eps ** 2 * n ** 2)) * alpha2 * alpha1
        a[0, 2] = (alpha2 / (n * sig_eps ** 2))
        a[1, 2] = (alpha1 / (n * sig_eps ** 2))
        a[2, 0] = (alpha2 / (n * sig_eps ** 2))
        a[2, 1] = (alpha1 / (n * sig_eps ** 2))

        opt_params = solve_3_eqs(a, b)

        exp_err = exp_val - (opt_params[0] * alpha2 / n) - (
            opt_params[1] * alpha1 / n) - opt_params[2]

        f = self.logp_params(opt_params)
        f += normal_loglike(exp_err, mu=0, sig=sig_eps)
        return opt_params, f


class BlackBoxBackCalc(BaseBackCalculator):
    """
    Base class for any black box back calculator in the EISD scheme
    """

    def __init__(self, dtype):
        super(BlackBoxBackCalc, self).__init__(nparams=0, dtype=dtype)

    def get_default_params(self):
        """
        No nuisance parameters required for this model. See BaseBackCalculator
        for more info

        :return:
        """
        return []

    def get_random_params(self):
        """
        No nuisance parameters required for this model. See BaseBackCalculator
        for more info

        :return:
        """
        return []

    def get_random_err(self, data_id):
        """
        Return a random value from the normal distribution corresponding
        to the error std value for this data id. See BaseBackCalculator
        for more info

        :param data_id:
        :return:
        """
        err_sig = self.get_err_sig(data_id)
        return np.random.normal(loc=0, scale=err_sig)

    def get_err_sig(self, data_id):
        """
        Get the std of error corresponding to this data id. See
        BaseBackCalculator for more info

        :param data_id:
        :return:
        """
        raise NotImplementedError

    def logp_params(self, params=None):
        """
        No nuisance parameters required for this model. See BaseBackCalculator
        for more info

        :param params:
        :return:
        """
        return 0.0

    def get_default_err(self, data_id=None):
        """
        Default is 0.0. See BaseBackCalculator for more info

        :param data_id:
        :return:
        """
        return 0.0

    def logp_err(self, err, data_id):
        """
        Error represented as gaussian with std from get_err_sig
        See BaseBackCalculator for more info

        :param err:
        :param data_id:
        :return:
        """
        sig_err = self.get_err_sig(data_id)
        logp = normal_loglike(err, mu=0, sig=sig_err)
        return logp

    def back_calc(self, xi, params=None):
        """
        this must be implemented for any new method based on the
        DataID structure for that back-calculator

        :param xi:
        :param params:
        :return:
        """
        raise NotImplementedError

    def get_default_struct_val(self):
        """
        Return a shift id for the first measurement of the first
        structure

        :return:
        """
        return Measurement(data_id=None, val=None)

    def calc_opt_params(self, n, beta_sig, sig_eps, exp_val):
        """
        Analytically calculate the optimal back-calculation error.
        For more info see BaseBackCalculator

        :param n: number of structures
        :param beta_sig: [sum of shift back calculations, std for this atom]
        :param sig_eps:
        :param exp_val:
        :return: optimal back calculator error in a list
        """
        beta = beta_sig[0] / n
        sig_bc = beta_sig[1]
        alpha = (sig_bc ** 2 / sig_eps ** 2)
        eps_back_opt = (alpha * (beta - exp_val)) / (1 + alpha)

        f = normal_loglike(eps_back_opt, mu=0, sig=sig_bc)
        exp_err = beta - exp_val - eps_back_opt
        f += normal_loglike(exp_err, mu=0, sig=sig_eps)
        return [eps_back_opt], f


class ShiftBackCalc(BlackBoxBackCalc):
    """
    Back calculator for chemical shifts. Uses SHIFTX2 for the back-calculation,
    which has no nuisance parameters.
    """

    def __init__(self):
        super(ShiftBackCalc, self).__init__("shift")

    def get_err_sig(self, data_id):
        """
        Get the std corresponding to the SHIFTX2 rmsd for this data id

        :param data_id:
        :return:
        """
        atom = data_id.atom_
        sig_err = SHIFTX2_RMSD[atom]
        return sig_err

    def back_calc(self, xi, params=None):
        """
        In this case, xi contains the back-calculated measurement.
        See BaseBackCalculator for more info.

        :param xi:
        :param params:
        :return:
        """
        return xi.val_


class RDCBackCalc(BlackBoxBackCalc):
    """
    Back calculator for residual dipolar couplings. Assumes that an RDC back-calculator
    has been run and read in by Structure (same as it is done for ShiftBackCalc
    :param err_std: I don't know the std of error for these, so it must be input for now
    """

    def __init__(self, err_std):
        self.errSig_ = err_std
        super(RDCBackCalc, self).__init__("rdc")

    def get_err_sig(self, data_id):
        """
        For now, return the member self.errSig_
        :param data_id:
        :return:
        """
        return self.errSig_

    def back_calc(self, xi, params=None):
        """
        xi should contain the back-calculated value in xi.val_
        :param xi:
        :param params:
        :return:
        """
        return xi.val_


class RHBackCalc(BlackBoxBackCalc):
    """
    Back calculator for hydrodynamic radius. HYDROPRO should have been run before
    and the results should be stored in Structure. I don't know the std of error
    for these, so it must be input for now
    """

    def __init__(self, err_std):
        self.errSig_ = err_std
        super(RHBackCalc, self).__init__("rh")

    def get_err_sig(self, data_id):
        """
        For now, return the member self.errSig_
        :param data_id:
        :return:
        """
        return self.errSig_

    def back_calc(self, xi, params=None):
        """
        xi should contain the back-calculated value in xi.val_
        :param xi:
        :param params:
        :return:
        """
        return xi.val_


class SAXSBackCalc(BlackBoxBackCalc):
    """
    Back calculator for SAXS data. CRYSOL should have already been run
    and the results should be stored in Structure. I don't know the std
    of those, so it must be input for now
    """

    def __init__(self, err_std):
        self.errSig_ = err_std
        super(SAXSBackCalc, self).__init__("saxs")

    def get_err_sig(self, data_id):
        """
        For now, return the member self.errSig_
        :param data_id:
        :return:
        """
        return self.errSig_

    def back_calc(self, xi, params=None):
        """
        xi should contain the back-calculated value in xi.val_
        :param xi:
        :param params:
        :return:
        """
        return xi.val_


class NOEBackCalc(BaseBackCalculator):
    """
    Back calculator for NOE restraints. This uses the isolated spin pair
    approximation, with parameter gamma
        NOE = gamma * r ^ -6
    where r is the distance between two atoms
    :param gamma_mu: mean of gamma parameter distribution
    :param gamma_std: std of gamma parameter distribtuion
    :param err_sig: std of errors (default is 0, b/c error should be
                    encapsulated by gamma distribution)
    """

    def __init__(self, gamma_mu, gamma_std, err_sig=0):
        self.gmu_ = gamma_mu
        self.gstd_ = gamma_std
        self.errSig_ = err_sig
        super(NOEBackCalc, self).__init__(1, "noe")

    def get_err_sig(self, data_id=None):
        """
        See BaseBackCalculator for more info

        :param data_id:
        :return:
        """
        return self.errSig_

    def get_random_params(self):
        """
        See BaseBackCalculator for more info

        :return:
        """
        params = [np.random.normal(loc=self.gmu_, scale=self.gstd_)]
        return params

    def logp_params(self, params):
        """
        Parameters are represented as normals. See BaseBackCalculator for more
        info

        :param params:
        :return:
        """
        logp = normal_loglike(params[0], mu=self.gmu_, sig=self.gstd_)
        return logp

    def get_random_err(self, data_id=None):
        """
        data_id is optional for this back-calculator. See BaseBackCalculator for
        more info

        :param data_id:
        """
        return np.random.normal(loc=0, scale=self.errSig_)

    def get_default_params(self):
        """
        Return mean of param values. See BaseBackCalculator for more info

        :return:
        """
        return [self.gmu_]

    def get_default_err(self, data_id=None):
        """
        Default if mean of error distribution See BaseBackCalculator for more
        info

        :param data_id:
        :return:
        """
        return 0

    def logp_err(self, err, data_id=None):
        """
        Error is represemted as a normal. See BaseBackCalculator for more info

        :param err:
        :param data_id:
        """
        logp = normal_loglike(err, mu=0, sig=self.errSig_)
        return logp

    def back_calc(self, xi, back_params):
        """
        Implementation of the ISPA equation. See BaseBackCalculator for more
        info

        :param xi:
        :param back_params:
        """
        r = xi.val_
        gamma = back_params[0]
        n = gamma * (r ** (-6))
        return n

    def get_default_struct_val(self):
        """
        Default is [pi, pi] dihedral angle. See BaseBackCalculator for more info
        :return:
        """
        meas = Measurement(data_id=None,
                           val=1)
        return meas

    def calc_opt_params(self, n, alpha, sig_eps, exp_val):
        """
        Analytically calculate the optimal gamma value
        For more info see BaseBackCalculator
        :param n: number of structures
        :param alpha: pre-calculated sum of r^-6 values
        :param sig_eps: experimental error
        :param exp_val: experimental data point
        :return: optimal gamma in a one-element list and the log likelihood of the result
        """
        alpha = alpha[0]
        num = n ** 3 * sig_eps ** 4 * self.gmu_
        num += n ** 2 * sig_eps ** 2 * self.gstd_ ** 2 * alpha * exp_val

        den = n ** 3 * sig_eps ** 4
        den += n * sig_eps ** 2 * self.gstd_ ** 2 * alpha ** 2

        opt_params = [num / den]
        exp_err = exp_val - ((opt_params[0] * alpha) / n)

        f = self.logp_params(opt_params)
        f += normal_loglike(exp_err, mu=0, sig=sig_eps)
        return opt_params, f
