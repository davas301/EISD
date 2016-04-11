import os

import numpy as np

import readutil
from backcalc import JCoupBackCalc, ShiftBackCalc
from eisd import DataEISD, EISDOPT
from priors import UniformPrior
# from priors2 import En
from structure import Structure

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
Module for testing the program
Below are example programs
"""


def test_ensemble(pdbdir):
    """
    Example program showing how to calculate the probability of an ensemble
    with some EISD and prior
    :param pdbdir: directory of pdb files for this ensemble
    :return: probability of ensemble
    """
    all_paths = [os.path.join(pdbdir, f) for f in os.listdir(pdbdir) if
                 ".pdb" in f and ".cs" not in f]
    all_paths = np.random.choice(all_paths, size=100)

    # build all the structures:
    structs = [Structure(f) for f in all_paths]

    # get data: (example is j-coupling data for AB42)
    exp_data = readutil.get_ab42_jcoup_data()

    # build back-calculator and  data eisd calculator:
    back_calc = JCoupBackCalc()
    # don't consider back-calc error for j-coupling (error is in nuisance
    # parameter variability):
    data_eisd = DataEISD(back_calc, exp_data, no_bc_err=True)

    # build prior:
    prior = UniformPrior(10)  # say there are possible 10 ensembles

    # calculate log probability:
    prior_logp = prior.calc_prior_logp()
    data_logp = data_eisd.calc_logp(structs)
    logp = prior_logp + data_logp

    outstr = "The ensemble at %s has a prior log probability of %.3f, " \
             "a fit-to-data log probability of %.3f and a total log " \
             "probability of %.3f" % (
                 pdbdir, prior_logp, data_logp, logp)
    print outstr
    return logp


def optimize_ensemble():
    """
    Example program showing how to find the optimal subset of an ensemble
    """
    pdbdir = "/Users/davidbrookes/data/pdb_J2_2"
    subset_size = 1000

    # build prior
    # pdb_energy_dict = readutil.get_md_energies()
    # ref_energies = pdb_energy_dict.values()
    prior = UniformPrior(1)

    # build DataEISD objects
    jcoup_eisd = DataEISD(JCoupBackCalc(), readutil.get_ab42_jcoup_data(),
                          no_bc_err=True, no_opt=True)
    shift_eisd = DataEISD(ShiftBackCalc(), readutil.get_ab42_shift_data(),
                          no_opt=True)
    data_eisds = [jcoup_eisd, shift_eisd]

    # define a cooling schedule for simulation annealing (same as default here):
    def cool_sched(t):
        """
        Cooling schedule
        :param t: fraction of iterations
        :return: "Temperature" for simulated annealing
        """
        return np.exp(-(2 * t) ** 2)

    savefile = "./output/md_eisd_opt.txt"
    stats_file = "./output/md_eisd_opt_stats.txt"

    optimizer = EISDOPT(pdbdir, prior, data_eisds, savefile,
                        subsize=subset_size, verbose=True,
                        stats_file=stats_file)
    niter = int(1e5)
    optimizer.opt(niter, cool_sched=cool_sched)




test_ensemble("/Users/davidbrookes/data/pdb_J2_2")

