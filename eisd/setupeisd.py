import os
import sys
from backcalc import JCoupBackCalc, ShiftBackCalc
from eisdcore import DataEISD, EISDOPT
from priors import UniformPrior
from readutil import read_chemshift_data, read_jcoup_data
from util import GaussianCoolSched, LinearCoolSched, str_to_bool, read_pdb_list

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
__date__ = '7/7/16'


class InputFile(object):
    """
    Class for reading EISD input files
    :param path: path to input file
    """

    def __init__(self, path):
        """
        Reads program parameters in the input file
        """
        self.path_ = path

        # keywords. Required if initialized by None
        self.keys_ = {
            # General parameters:
            'PDBLIST': None,  # file containing list of pdb files
            'SUB_SIZE': None,  # size of final subset

            # Data parameters:
            'USE_JCOUP': "0",  # use JCoupling data?
            'JCOUP_PATH': None,  # path to Jcoupling data
            'USE_SHIFT': "0",  # use chemical shift data?
            'SHIFT_PATH': None,  # path to chemical shift data
            'RUN_SHIFTX': "0",  # run shiftx on every structure?
            'SHIFTX_EXE': None,  # path to shiftx executable

            # Simulated annealing (SA) parameters
            'N_ITER': 10000,  # number of SA iterations
            'COOL_SCHED': 'gaussian',  # type of SA cooling schedule
            # Gaussian cooling schedule is: T = t0 * exp(-(scale*t)**2)
            # Linear cooling schedule is: T = t0 - scale*t
            'COOL_T0': None,  # starting temperature for cooling
            'COOL_SCALE': None,  # scale for cooling
            'SAVE_FILE': None,  # save file for list of files in final subset
            'STATS_FILE': None,  # save stats of optimization

            # Prior parameters:
            'PRIOR': 'uniform',
            'PRIOR_UNI_M': 10,  # size of ensemble state space for uniform prior
        }

        self._read_file()
        valid = self._check_params()
        if not valid:
            print "Input file is not valid. Aborting program."
            sys.exit()

    def _find_keyword(self, line):
        """
        Find and set values for keywords
        :param line: the line to search for a keyword
        """
        split = line.split()
        if len(split) == 0:
            return
        if split[0] in self.keys_.keys():
            if len(split) == 1:
                return
            self.keys_[split[0]] = split[1]
        else:
            print "%s is not a valid keyword. Continuing anyway..." % split[0]

    def _read_file(self):
        """
        Read the input file and set keywords
        """
        f = open(self.path_)
        lines = f.readlines()
        f.close()
        for line in lines:
            if line.startswith("#"):
                continue
            self._find_keyword(line)

    def _check_params(self):
        """
        Check to make sure the input valid has a valid parameter list
        :return: a boolean stating whether the input file is valid
        """

        # first check required inputs
        if self.keys_['PDBLIST'] is None:
            print "PDBLIST input is required and was not found"
            return False
        if self.keys_['SUB_SIZE'] is None:
            print "SUB_SIZE input is required and was not found"
            return False
        if self.keys_['SAVE_FILE'] is None:
            print "SAVE_FILE input is required and was not found"
            return False

        # now make sure things are the correct type:
        for bool_key in ['USE_JCOUP', 'USE_SHIFT', 'RUN_SHIFTX']:
            try:
                self.keys_[bool_key] = str_to_bool(self.keys_[bool_key])
            except ValueError:
                print "%s is not a valid value for %s. Must be a boolean" % (str(
                    self.keys_[bool_key]), bool_key)
                return False

        for int_key in ['SUB_SIZE', 'N_ITER', 'PRIOR_UNI_M']:
            try:
                self.keys_[int_key] = int(self.keys_[int_key])
            except ValueError:
                print "%s is not a valid value for %s. Must be integer" % (str(
                    self.keys_[int_key]), int_key)
                return False

        # Now make sure paths exist:
        if self.keys_['USE_JCOUP']:
            if not os.path.exists(self.keys_['JCOUP_PATH']):
                print "%s is not a valid path to a j coupling file" % \
                      self.keys_['JCOUP_PATH']
                return False

        if self.keys_['USE_SHIFT']:
            if not os.path.exists(self.keys_['SHIFT_PATH']):
                print "%s is not a valid path to a chemical shift file" % \
                      self.keys_['SHIFT_PATH']
                return False
            if self.keys_['RUN_SHIFTX']:
                if not os.path.exists(self.keys_['SHIFTX_EXE']):
                    print "%s is not a valid path to a SHIFTX2 executable" % \
                          self.keys_['RUN_SHIFTX']
                return False

        # check cooling schedule parameters:
        default_cool_params = {
            'gaussian': [1, 2],
            'linear': [self.keys_['N_ITER'] + 1, 1]
        }
        cool_scheds = ['gaussian', 'linear']
        if self.keys_['COOL_SCHED'] not in cool_scheds:
            print "%s is not a valid cooling schedule. Please input " \
                  "'gaussian, or 'linear'"
            return False
        else:

            for cs in cool_scheds:
                if self.keys_['COOL_SCHED'] == cs:
                    if self.keys_['COOL_T0'] is None:
                        self.keys_['COOL_T0'] = default_cool_params[cs][0]
                    else:
                        try:
                            self.keys_['COOL_T0'] = float(self.keys_['COOL_T0'])
                        except ValueError:
                            print "%s is not a valid starting temperature. " \
                                  "Please input a float."
                            return False
                    if self.keys_['COOL_SCALE'] is None:
                        self.keys_['COOL_SCALE'] = default_cool_params[cs][1]
                    else:
                        try:
                            self.keys_['COOL_SCALE'] = float(
                                self.keys_['COOL_SCALE'])
                        except ValueError:
                            print "%s is not a valid cooling scale. " \
                                  "Please input a float."
                            return False

        priors = ['uniform']
        if self.keys_['PRIOR'] not in priors:
            print "%s is not a valid prior. Please input 'uniform' " \
                  "or ... thats it"
            return False
        return True


class SetupOpt(object):
    """
    Given an input file object, set up an optimization run
    :param inp_file: an InputFile object
    """

    def __init__(self, inp_file):
        self.inpFile_ = inp_file
        # self.pdbdir_ = inp_file.keys['PDBDIR']
        # self.subsize_ = inp_file.keys['SUBSIZE']

        self.prior_ = self._setup_prior()
        self.dataEISD_ = self._setup_eisd_obj()
        self.opt_ = self._setup_opt()

    def _setup_prior(self):
        """
        Set up the prior distribution
        :return: a BasePrior object
        """
        prior = None
        if self.inpFile_.keys_['PRIOR'] == 'uniform':
            prior = UniformPrior(self.inpFile_.keys_['PRIOR_UNI_M'])
        return prior

    def _setup_eisd_obj(self):
        """
        Setup the DataEISD object that will be used
        :return: DataEISD object
        """
        back_calcs = []
        data_dicts = []
        if self.inpFile_.keys_['USE_JCOUP']:
            back_calcs.append(JCoupBackCalc())
            data_file = read_jcoup_data(self.inpFile_.keys_['JCOUP_PATH'])
            data_dicts.append(data_file)
        if self.inpFile_.keys_['USE_SHIFT']:
            back_calcs.append(ShiftBackCalc())
            data_file = read_chemshift_data(self.inpFile_.keys_['SHIFT_PATH'])
            data_dicts.append(data_file)
        eisd = DataEISD(back_calcs, data_dicts)
        return eisd

    def _setup_opt(self):
        """
        Setup the optimizer object
        :return: an EISDOPT object
        """
        pdblist = read_pdb_list(self.inpFile_.keys_['PDBLIST'])
        subsize = self.inpFile_.keys_['SUB_SIZE']
        runshiftx = self.inpFile_.keys_['RUN_SHIFTX']
        savefile = self.inpFile_.keys_['SAVE_FILE']
        statsfile = self.inpFile_.keys_['STATS_FILE']

        if self.inpFile_.keys_['COOL_SCHED'] == 'gaussian':
            cool_sched = GaussianCoolSched(self.inpFile_.keys_['COOL_T0'],
                                           self.inpFile_.keys_['COOL_SCALE'])
        elif self.inpFile_.keys_['COOL_SCHED'] == 'linear':
            cool_sched = LinearCoolSched(self.inpFile_.keys_['COOL_T0'],
                                         self.inpFile_.keys_['COOL_SCALE'])
        else:
            cool_sched = None

        optimizer = EISDOPT(pdblist, self.prior_, self.dataEISD_, savefile,
                            subsize=subsize, verbose=True, stats_file=statsfile,
                            run_shiftx=runshiftx, cool_sched=cool_sched)
        return optimizer

    def run(self):
        """
        Runs the full setup optimization
        """
        niter = self.inpFile_.keys_['N_ITER']
        self.opt_.opt(niter)
