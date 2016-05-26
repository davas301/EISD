import os
import time
import numpy as np
import scipy.optimize as opt
from structure import Structure
from util import normal_loglike
from readutil import read_opt_out_file

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
Module for performing EISD calculation
"""


class DataEISD(object):
    """
    Implementation of the Experimental Inferential Structure Determination
    model. This does not include prior probabilities (priors should be
    calculated separately). Each DataEISD object calculates probabilities
    for one data type

    :param back_calc: BaseBackCalculator object
    :param exp_data: experimental data corresponding to back-calc
    :param no_exp_err: True if experimental error should not contribute
    :param no_params: True if parameter probabilities should not contribute
    :param no_bc_err: True if back-calculation error should not contribute
    :param no_opt: True if no optimization should be performed--just calculate
    error probability from normal
    """

    def __init__(self, back_calc, exp_data,
                 no_exp_err=False, no_params=False,
                 no_bc_err=False, no_opt=False):
        self.backCalc_ = back_calc

        self.M_ = len(exp_data)
        exp_tups = [(did, val) for did, val in exp_data.items()]
        self.expKeys_ = [e[0] for e in exp_tups]
        exp_vals = [e[1] for e in exp_tups]
        self.D_ = [float(v[0]) for v in exp_vals]
        self.expSigs_ = [float(v[1]) for v in exp_vals]
        self.bcSigs_ = [self.backCalc_.get_err_sig(did) for
                        did in self.expKeys_]

        self.noExpErr_ = no_exp_err
        self.noParams_ = no_params
        self.noBackErr_ = no_bc_err
        self.noOpt_ = no_opt

        self.lastOptParams_ = [None] * self.M_

    def get_all_struct_measures(self, structs, j):
        """
        Get all structural measurements for data point j
        :param structs: list of structures
        :param j: index of data point
        :return: list of Measurement objects
        """
        lout = [si.get_struct_measure(self.expKeys_[j]) for si in structs]
        return lout

    def compute_all_back_calc(self, structs, bc_params, j):
        """
        Perform back-calculation on every input structure for data point j

        :param structs: list of Structure objects
        :param bc_params: input parameters for back-calculator
        :param j: index of data point to back-calculate for
        :return: vector of back-calculations
        """
        n = len(structs)
        bc_vec = np.zeros(n)
        for i in range(n):
            struct_val = structs[i].get_struct_measure(self.expKeys_[j])
            bc_vec[i] = self.backCalc_.back_calc(struct_val, bc_params)
        return bc_vec

    def compute_back_calc_mean(self, structs, bc_params, j):
        """
        Compute mean of back-calculation on input structures for data point j

        :param structs: list of Structure objects
        :param bc_params: input parameters for back-calculator
        :param j: index of data point to back-calculate for
        :return: mean of back-calculations
        """
        return np.mean(self.compute_all_back_calc(structs, bc_params, j))

    def _logp_params(self, value):
        """
        Calculate the log probability of back-calculator parameters

        :param value: back-calculator parameters
        :return: log probability of parameters
        """
        return self.backCalc_.logp_params(value)

    def _random_params(self):
        """
        See BaseBackCalculator for more info

        :return: random parameter set
        """
        return self.backCalc_.get_random_params()

    def _logp_total_err(self, value, j):
        """
        For the no-opt sceneario. Return log probability of an error
        at data point j assuming the error distribution is a normal with
        variance equal to the sum of the experimental and back-calculator
        error variances

        :param value: value of error
        :param j: index of data point
        :return: log probability
        """
        return normal_loglike(value, mu=0, sig=np.sqrt(
            self.expSigs_[j] + self.bcSigs_[j]))

    def _logp_exp_err(self, value, j):
        """
        Log probability of experimental error

        :param value: error value
        :param j: index of experimental data point
        :return: log probability
        """
        return normal_loglike(value, mu=0, sig=self.expSigs_[j])

    def _logp_bc_err(self, value, j):
        """
        Log probability of back-calculation error

        :param value: value of error
        :param j: index of data point
        :return: log probability
        """
        return normal_loglike(value, mu=0, sig=self.bcSigs_[j])

    def _random_exp_err(self, j):
        """
        Return a random experimental error

        :param j: index of experimental data point
        :return: random error
        """
        return np.random.normal(loc=0, scale=self.expSigs_[j])

    def _random_bc_err(self, j):
        """
        Return a random back-calculator error

        :param j: index of data point
        :return: random error
        """
        return np.random.normal(loc=0, scale=self.bcSigs_[j])

    def _eval(self, structs, params, bc_err, j):
        """
        Given a list of structures,  bc_err, params and a data point index,
        return the experimental error at the data point. This essentially
        implements equation 5 in Brookes (2016).

        :param structs: list of Structure objects
        :param params: back-calc parameters
        :param bc_err: back-calculation error
        :param j: index of data point
        :return: experimental error
        """
        bc_val = self.compute_back_calc_mean(structs, params, j)
        exp_err = self.D_[j] - (bc_val - bc_err)
        return exp_err

    def calc_logp(self, structs):
        """
        Calculate the full EISD log probability given an ensemble
        :param structs: list of Structure objects
        :return: log probability
        """
        logp_total = 0
        params = self.backCalc_.get_default_params()
        for j in range(self.M_):
            if not self.noOpt_:
                # def calc_logp_j(x):
                #     """
                #     Calculate the logp of a single data point. Used
                #     as input to scipy optimization
                #
                #     :param x: inputs
                #     :return: -logp ( because max(f)=-min(-f) )
                #     """
                #     # start = time.time()
                #     for i in range(len(x)):
                #         if np.isnan(x[i]) or np.isinf(x[i]):
                #             return np.inf
                #     _params = x[:self.backCalc_.nParams_]
                #     if not self.noBackErr_:
                #         bc_err = x[-1]
                #     else:
                #         bc_err = 0
                #
                #     exp_err = self._eval(structs, _params, bc_err, j)
                #     _logp_j = self._logp_exp_err(exp_err, j)
                #     if not self.noBackErr_:
                #         _logp_j += self._logp_bc_err(bc_err, j)
                #     _logp_j += self._logp_params(_params)
                #     # print time.time() - start
                #     return -_logp_j
                #
                # if self.lastOptParams_[j] is None:
                #     x0_j = self._random_params()
                #     if not self.noBackErr_:
                #         x0_j.append(self._random_bc_err(j))
                # else:
                #     x0_j = self.lastOptParams_[j]
                # #
                # # start = time.time()
                # opt_result = opt.minimize(calc_logp_j, x0_j)
                # print opt_result.x, opt_result.fun
                # # print opt_result.nfev
                # print "Opt time: %f" % (time.time() - start)
                # print opt_result.x
                #
                # start = time.time()
                opt_params, f = self.backCalc_.calc_opt_params(
                    self.get_all_struct_measures(structs, j),
                    self.expSigs_[j], self.D_[j])
                # print opt_params, f

                # logp_opt_j = -opt_result.fun
                logp_opt_j = f
                logp_total += logp_opt_j
                # print

            else:
                bc_val = self.compute_back_calc_mean(structs, params, j)
                err = self.D_[j] - bc_val
                logp_j = self._logp_total_err(err, j)
                logp_total += logp_j

        return logp_total


class EISDOPT(object):
    """
    Class for optimizing an ensemble based on EISD probabilities

    :param pdb_dir: path to full ensemble of pdb structures
    :param prior: a BasePrior object
    :param data_eisds: list of DataEISD objects
    :param savefile: file to save best ensemble to (list of pdb names)
    :param subsize: size of sub-ensemble to optimize
    :param verbose: if True, will print updates
    :param stats_file: file to save statistics to (optional)
    """

    def __init__(self, pdb_dir, prior, data_eisds, savefile, subsize=1000,
                 verbose=True, stats_file=None, restartfile=None):
        self.pdbDir_ = pdb_dir
        self.allPDB_ = [f for f in os.listdir(self.pdbDir_) if
                        ".pdb" in f and ".cs" not in f]

        self.saveFile_ = savefile
        self.verbose_ = verbose
        self.fstats_ = stats_file

        self.Nsub_ = subsize
        self.prior_ = prior
        self.dataEISDs_ = data_eisds

        build_tup = self._build_start_set(restartfile=restartfile)
        self.stateFiles_, self.stateStructs_, self.startIter_ = build_tup
        self.priorArgs_ = [self.prior_.get_arg(s) for s in self.stateStructs_]
        # print self.priorArgs_

        # save last removed things for restoration:
        self.lastRemoveFile_ = None
        self.lastRemoveStruct_ = None
        self.lastRemovePrior_ = None

    def _build_start_set(self, restartfile=None):
        """
        Build a random start set of structures

        :return: list of files and list of Structure objects
        """
        if restartfile is None:
            state_files = list(np.random.choice(self.allPDB_, size=self.Nsub_,
                                                replace=False))
            start_iter = 0
        else:
            state_files, start_iter = read_opt_out_file(restartfile)

        state_structs = [None] * self.Nsub_
        for i in range(len(state_files)):
            f = state_files[i]
            shiftx_file = os.path.join(self.pdbDir_, f + ".cs")
            state_structs[i] = Structure(os.path.join(self.pdbDir_, f),
                                         shiftxfile=shiftx_file)
            if self.verbose_:
                if i % 100 == 0 and i > 0:
                    print "Number of structures built: %i / % i" % (
                        i, self.Nsub_)

        return state_files, state_structs, start_iter

    def _perturb(self):
        """
        Randomly remove a structure and add a random structure from the
        reservoir
        """
        rand_remove_idx = np.random.randint(0, self.Nsub_)
        self.lastRemoveFile_ = self.stateFiles_.pop(rand_remove_idx)
        self.lastRemoveStruct_ = self.stateStructs_.pop(rand_remove_idx)
        self.lastRemovePrior_ = self.priorArgs_.pop(rand_remove_idx)

        rand_add_file = np.random.choice(self.allPDB_)
        while rand_add_file in self.stateFiles_:
            rand_add_file = np.random.choice(self.allPDB_)

        rand_cs_file = os.path.join(self.pdbDir_, rand_add_file + ".cs")
        rand_add_struct = Structure(os.path.join(self.pdbDir_, rand_add_file),
                                    shiftxfile=rand_cs_file)

        self.stateFiles_.append(rand_add_file)
        self.stateStructs_.append(rand_add_struct)
        self.priorArgs_.append(self.prior_.get_arg(rand_add_struct))

    def _restore(self):
        """
        Restore the system to its state before the last perturbation
        """
        self.stateFiles_[-1] = self.lastRemoveFile_
        self.stateStructs_[-1] = self.lastRemoveStruct_
        self.priorArgs_[-1] = self.lastRemovePrior_

        self.lastRemoveFile_ = None
        self.lastRemoveStruct_ = None
        self.lastRemovePrior_ = None

    def _calc_data_prob(self):
        """
        Calculate the probability of the data for the current set of structures

        :return: probability of data
        """
        e = 0
        indi = [0] * len(self.dataEISDs_)
        for i in range(len(self.dataEISDs_)):
            start = time.time()
            ei = self.dataEISDs_[i].calc_logp(self.stateStructs_)
            print time.time() - start
            indi[i] = ei
            e += ei
        print
        return e, indi

    @staticmethod
    def default_cool(t):
        """
        Default cooling schedule for simulated annealing

        :param t: fraction of iterations
        :return: "Temperature" for simulated annealing
        """
        return np.exp(-(2 * t) ** 2)

    def opt(self, niter, cool_sched=None):
        """
        Perform simulated annealing procedure to find the subset of structures
        that maximimizes probability. Requires the number of iterations and
        the cooling schedule, which is a callable function that takes the
        current fraction of iterations performed

        :param niter: number of iterations
        :param cool_sched: callable cooling schedule function
        """
        # default cooling schedule is gaussian:
        if cool_sched is None:
            cool_sched = EISDOPT.default_cool

        stats_str = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%ss" % (
            "iter", "accept?", "e", "prior", "jcoup_e", "shift_e",
            "p_a", "iter_time")
        if self.verbose_:
            print stats_str

        if self.fstats_ is not None:
            statsf = open(self.fstats_, 'w+')
            statsf.write(stats_str + "\n")

        # determine starting energy and prior scale parameter
        prior0 = self.prior_.calc_prior_logp(self.priorArgs_)
        data0, indi0 = self._calc_data_prob()
        theta = 0.05 * (data0 / prior0)  # scale prior by theta
        e0 = data0 + theta * prior0

        abs_min = [e0, self.stateFiles_]
        for i in range(self.startIter_, niter):
            start = time.time()  # keep track of step time
            self._perturb()

            data1, indi1 = self._calc_data_prob()
            prior1 = self.prior_.calc_prior_logp(self.priorArgs_)
            # if prior1 < 0:
            #     self._restore()
            #     continue
            e1 = data1 + theta * prior1

            dE = e0 - e1
            accept = False
            if dE < 0:
                p_a = 1
                accept = True
            else:
                ti = float(i) / niter
                T = cool_sched(ti)
                p_a = np.exp(-dE / T)
                rand = np.random.rand()
                if rand < p_a:
                    accept = True

            did_accept = "-"
            if accept:
                e0 = e1
                did_accept = "+"

                # save if absolute minimum
                if e0 - abs_min[0] > 0:
                    abs_min[0] = e0
                    abs_min[1] = self.stateFiles_
                    outf = open(self.saveFile_, "w+")
                    outf.write('iter: %i\n' % i)
                    for currf in self.stateFiles_:
                        outf.write("%s\n" % currf)
            else:
                self._restore()

            stats_str = "%i\t%s\t%f\t%f\t%f\t%f\t%f\t%fs" % (
                i, did_accept, e1, prior1, indi1[0], indi1[1], p_a,
                time.time() - start
            )

            if self.verbose_:
                print stats_str
            if self.fstats_ is not None:
                statsf.write(stats_str + "\n")
