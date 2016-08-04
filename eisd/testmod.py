import os
import numpy as np
import readutil
from backcalc import JCoupBackCalc, ShiftBackCalc
from eisdcore import DataEISD, EISDOPT
from priors import UniformPrior, EnergyPrior
from eisdstructure import Structure
import matplotlib.pyplot as plt
import time
import util

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
    # all_paths = np.random.choice(all_paths, size=100)

    # build all the structures:
    structs = []
    i = 0
    for f in all_paths:
        shiftx_file = os.path.join(pdbdir, f + ".cs")
        structs.append(Structure(f, shiftxfile=shiftx_file))
        i += 1
        if i % 100 == 0:
            print "Numer of structures built: %i / %i" % (i, len(all_paths))

    jcoup_eisd = DataEISD([JCoupBackCalc()], [readutil.get_ab42_jcoup_data()])
    shift_eisd = DataEISD([ShiftBackCalc()], [readutil.get_ab42_shift_data()])

    # build prior:
    prior = UniformPrior(10)  # say there are possible 10 ensembles

    jcoup_eisd.set_struct_vals(structs)
    shift_eisd.set_struct_vals(structs)

    # calculate log probability:
    prior_logp = prior.calc_prior_logp()
    start = time.time()
    jcoup_logp = jcoup_eisd.calc_logp()
    shift_logp = shift_eisd.calc_logp()
    data_logp = jcoup_logp + shift_logp
    print time.time() - start, jcoup_logp + shift_logp
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
    # pdbdir = "/Users/davidbrookes/data/pdb_J2_2"
    pdbdir = "/Users/davidbrookes/misc/ab42_ens/"
    subset_size = 100

    # pdb_dict = readutil.get_md_energies()
    # ref_array = pdb_dict.values()
    # prior = EnergyPrior(ref_array, pdb_dict=pdb_dict)
    prior = UniformPrior(10)

    # build DataEISD objects
    # jcoup_eisd = DataEISD(JCoupBackCalc(), readutil.get_ab42_jcoup_data(),
    #                       no_bc_err=True)
    # shift_eisd = DataEISD(ShiftBackCalc(), readutil.get_ab42_shift_data())
    # data_eisds = [jcoup_eisd, shift_eisd]
    data_eisds = DataEISD([JCoupBackCalc(), ShiftBackCalc()],
                          [readutil.get_ab42_jcoup_data(),
                           readutil.get_ab42_shift_data()])

    savefile = "../output/md_eisd_opt14.txt"
    stats_file = "../output/md_eisd_opt_stats14.txt"

    optimizer = EISDOPT(pdbdir, prior, data_eisds, savefile,
                        subsize=subset_size, verbose=True,
                        stats_file=stats_file)
    niter = int(5e5)
    optimizer.opt(niter)


def plot_opt_progress():
    """
    Plot optimization progress
    """
    paths = ["../output/old_opts/md_eisd_opt_stats7.txt",
             "../output/old_opts/md_eisd_opt_stats8.txt"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    seqs = []
    xs = []
    j = 0
    for path in paths:
        f = open(path)
        p = []
        x = []
        it = 0
        for line in f:
            if it == 0:
                it += 1
                continue
            split = line.split()
            if len(split) < 6:
                continue
            x.append(int(split[0]))
            if split[1] == "+":
                p.append(float(split[2]))
            else:
                p.append(p[-1])

        xs.append(x)
        seqs.append(p)
        ax.plot(x, p, label='Run %i Total' % j, alpha=0.5, lw=1)

        j += 1

    rs = []
    iters = []
    for i in range(1000, len(seqs[0]), 100):
        seqs_shortened = [seqs[k][:i] for k in range(len(seqs))]
        r = util.calc_psrf(seqs_shortened)
        rs.append(r)
        iters.append(i)

    # ax2 = ax.twinx()
    # ax2.plot(iters, rs, c='r', label='PSRF')
    # ax2.set_ylabel("Potential Scale Reduction Factor (PSRF)")
    # ax2.set_ylim(1, 3)
    ax.axhline(706.4041, c='k', ls='--', label='ENSEMBLE optimized ensemble')
    ax.set_xlim(-int(len(seqs[0]) / 20), int(len(seqs[0])))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("EISD Log Probability")
    handles1, labels1 = ax.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()
    # ax.legend(handles1+handles2, labels1+labels2, frameon=False,
    #           loc='lower right')
    ax.legend(frameon=False, loc='lower right')
    # plt.savefig("../output/opt_prog_100_jcoup_sh.pdf")
    plt.savefig("../output/opt_prog.png", dpi=300)
    plt.close()


def test_psrf():
    xs = np.linspace(1, 1000, 1000)
    m = 10
    seqs = [None] * m
    rs = []
    r_xs = []
    for x in xs:
        std = 1./(x**0.5)
        for j in range(m):
            if seqs[j] is None:
                seqs[j] = []
            seqs[j].append(np.random.normal(1, scale=std))
        if x > 100:
            r_xs.append(x)
            rs.append(util.calc_psrf(seqs))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for j in range(m):
        ax.plot(xs, seqs[j], alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(r_xs, rs, c='r', label='PSRF')
    ax2.set_ylabel("Potential Scale Reduction Factor (PSRF)")
    # ax2.set_ylim(1, 3)

    ax.set_xlim(min(xs), max(xs))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("EISD Log Probability")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1+handles2, labels1+labels2, frameon=False,
              loc='lower right')
    # plt.savefig("../output/opt_prog_100_jcoup_sh.pd f")
    plt.savefig("../output/psrf_test.png", dpi=300)
    plt.close()

# test_psrf()/
plot_opt_progress()
# optimize_ensemble()