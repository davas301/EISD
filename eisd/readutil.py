import glob
import os
import subprocess
import time

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
Module containing various useful classes and functions for external
operations (reading files, running external programs, etc.)
"""


class BaseDataID(object):
    """
    Base class for objects that contain unique IDs for data points. The goal
    of these classes is to ensure that experimental measurements are
    solidly connected to the corresponding structural and back-calculation
    measurements. These IDs must be hashable and have equality comparisons
    """

    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class ShiftID(BaseDataID):
    """
    Storage objects containing information that identifies a unique chemical
    shift. A shift is defined by a residue number and atom_ name
    """

    def __init__(self, res_num, atom_name):
        self.res_ = res_num
        self.atom_ = atom_name

    def __hash__(self):
        """
        Hash the tuple of the two members
        """
        return hash((self.res_, self.atom_))

    def __eq__(self, other):
        if other.res_ == self.res_ and other.atom_ == self.atom_:
            return True
        else:
            return False

    def __str__(self):
        s = "%i\t%s" % (self.res_, self.atom_)
        return s


class JCoupID(BaseDataID):
    """
    ID objects for j coupling values. Only requires a residue number
    """

    def __init__(self, res_num):
        self.res_ = res_num

    def __hash__(self):
        return hash(self.res_)

    def __eq__(self, other):
        if other.res_ == self.res_:
            return True
        else:
            return False

    def __str__(self):
        s = "%i" % self.res_
        return s


class Measurement(object):
    """
    Class for storing all of the info in a data measurement
    (experimental or structural). Contains a data ID and a value
    """

    def __init__(self, data_id=None, val=None):
        self.dataID_ = data_id
        self.val_ = val


class RunShiftX(object):
    """
    Class for running the SHIFTX2 command line program
    """
    DEFAULT_EXE = "/usr/local/bin/shiftx2/shiftx2.py"
    DEFAULT_TEMPDIR = "./tmp/"

    def __init__(self, exe=DEFAULT_EXE, tempdir=DEFAULT_TEMPDIR):
        """
        :param exe: path to SHIFTX2 executable
        :param tempdir: path to a directory to story temporary files
        """
        self.exe_ = exe
        self.tempdir_ = tempdir

    def run_shiftx_once(self, inpath, backbone_only=False,
                        clean=True, no_output=True):
        """
        Runs SHIFTX2 for a single input pdb file
        :param inpath: path to input pdb file
        :param backbone_only: True if SHIFTX2 should only run for backbone atoms
        :param clean: True if the temporary files should be deleted
        :param no_output: If true, no SHIFTX2 ouput will be shown on
        :return: path to output ".cs" file
        """
        args = ['python', self.exe_, '-i', inpath,
                '-z', self.tempdir_, '-c', 'A']
        if backbone_only:
            args += ['-a', 'BACKBONE']
        if no_output:
            fnull = open(os.devnull, 'w')
            subprocess.call(args, stdout=fnull, stderr=subprocess.STDOUT)
        else:
            subprocess.call(args)
        if clean:
            self._clean()
        return inpath + ".cs"

    def _clean(self):
        """
        Removes temporary files
        """
        tmp_files = glob.glob(self.tempdir_ + "*")
        for fn in tmp_files:
            subprocess.call(["rm", "-rf", fn])

    @staticmethod
    def read_ouput(f):
        """
        Reads a shiftx2 output file
        :param f: output file to be read
        :return: {ShiftID: shift_val} dict
        """
        lines = open(f).readlines()
        dout = {}
        for i in range(1, len(lines)):
            split = lines[i].split(',')
            res_num = int(split[0])
            # res_name = split[1]
            atom_name = split[2]
            shift_value = float(split[3])
            sid = ShiftID(res_num=res_num, atom_name=atom_name)
            dout[sid] = shift_value
        return dout


def write_dihed_to_file(structs, outname, verbose=True):
    """
    Write the dihedral angles of a list of structures into a tab-separated
    file where each line represents a single structure and the columns
    alternate phi, psi angles for each residue
    :param structs: list of Structure objects
    :param outname: path to where the dihed file should be written
    :param verbose: If True updates will be written to terminal
    """
    fout = open(outname, 'w+')
    i = 0
    start = time.time()
    if verbose:
        print "Writing dihedral angles to a file. This may take some time"
    for s in structs:
        all_dihed = s.get_all_dihed()
        phis = []
        psis = []
        for j in range(1, max(all_dihed.keys())):
            phis.append(all_dihed[j][0])
            psis.append(all_dihed[j][1])
        outstr = ''
        outstr += "%f\t" % psis[0]
        for j in range(1, len(phis) - 1):
            outstr += "%f\t%f\t" % (phis[j], psis[j])
        outstr += "%f\n" % psis[-1]
        fout.write(outstr)
        i += 1
        if verbose:
            if i % 100 == 0:
                print "Number of structures read: %i / %i in %fs" % \
                      (i, len(structs), time.time() - start)

    fout.close()


def read_dihed_file(path):
    """
    Read the dihedral angle file written by write_dihed_to_file
    :param path:
    :return: numpy array containg dihedral angles
    """
    f = open(path)
    lines = f.readlines()
    n = len(lines)

    ncol = len(lines[0].split())
    nrow = n
    dihed_array = np.zeros((nrow, ncol))
    idx = range(0, n)
    for i in range(nrow):
        split = lines[idx[i]].split()
        if len(split) < 80:
            continue
        for j in range(len(split)):
            dihed_array[i, j] = float(split[j])
    return dihed_array


def get_ab42_jcoup_data():
    """
    Read the J-coupling data for aB42
    :return: a {JCoupID: val} dict
    """
    jcoup_data_path = "../test_data/bax_2016_jcoup.txt"
    jcoup_data = {}
    expf = open(jcoup_data_path)
    lines = expf.readlines()
    expf.close()
    for line in lines[1:]:
        split = line.split()
        res_num = int(split[0])
        val = float(split[1])
        err = float(split[2])
        jid = JCoupID(res_num)
        jcoup_data[jid] = (val, err)
    return jcoup_data


def get_ab42_shift_data():
    """
    Read the shift data for aB42
    :return: a {ShiftID: val} dict
    """
    shift_data_path = "../test_data/filtered_ab42_cs.txt"
    shift_data = {}
    expf = open(shift_data_path)
    lines = expf.readlines()
    expf.close()
    for line in lines:
        split = line.split()
        res_num = int(split[0])
        atom = split[1]
        val = float(split[2])
        err = float(split[3])
        sid = ShiftID(res_num, atom)
        shift_data[sid] = (val, err)
    return shift_data


def get_md_energies():
    """
    Read the energy file containing all the energies for the
    MD ensemble
    :return: {filename: energy} dict
    """
    path = "test_data/MD_energies.txt"
    f = open(path)
    lines = f.readlines()
    f.close()
    edict = {}
    for line in lines:
        split = line.split()
        fname = split[0]
        # fnum = int(fname.split('.')[-1])
        energy = float(split[1])
        edict[fname] = energy
    return edict
