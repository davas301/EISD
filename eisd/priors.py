import numpy as np

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

    :param n: number of candidate ensembles

    """

    def __init__(self, n):
        self.n_ = n
        super(UniformPrior, self).__init__()

    def calc_prior_logp(self):
        """
        Probability is just 1/n. See BasePrior for more info

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
