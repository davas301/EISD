import glob
import os
import subprocess
import sys
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

    def __init__(self, dtype):
        self.dtype_ = dtype

    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class ShiftID(BaseDataID):
    """
    Storage objects containing information that identifies a unique chemical
    shift. A shift is defined by a residue number and atom name

    :param res_num: residue number of measurement
    :param atom_name: name of atom with residue
    """

    def __init__(self, res_num, atom_name):
        self.res_ = res_num
        self.atom_ = atom_name
        super(ShiftID, self).__init__("shift")

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

    :param res_num: residue number of measurement
    """

    def __init__(self, res_num):
        self.res_ = res_num
        super(JCoupID, self).__init__("jcoup")

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


class RDCID(BaseDataID):
    """
    ID objects for RDC values. Only requires a residue number (assumes they are
    intra-residue H-N RDCs)

    :param res_num: residue number of measurement
    """

    def __init__(self, res_num):
        self.res_ = res_num
        super(RDCID, self).__init__("rdc")

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


class RHID(BaseDataID):
    """
    ID objects for hydrodynamic radius values. There is only one
    hydrodynamic radius for a structure, so this does not require
    any parameters, but it can have an optional name
    """

    def __init__(self, name="rh"):
        self.name_ = name
        super(RHID, self).__init__("rh")

    def __hash__(self):
        return hash(self.name_)

    def __eq__(self, other):
        if other.name_ == self.name_:
            return True
        else:
            return False

    def __str__(self):
        s = "%s" % self.name_
        return s


class SAXSID(BaseDataID):
    """
    ID objects for SAXS back-calc values. Requires the
    radius defining the SAXS value
    """

    def __init__(self, radius):
        self.radius_ = radius
        super(SAXSID, self).__init__("saxs")

    def __hash__(self):
        return hash(self.radius_)

    def __eq__(self, other):
        if other.intensity_ == self.radius_:
            return True
        else:
            return False

    def __str__(self):
        s = "%f" % self.radius_
        return s


class NOEID(BaseDataID):
    """
    ID objects for NOE back-calc values. NOE values are calculated between
    two atoms, so this requires residue of first atom, first atom name,
    residue of second atom and second atom name
    :param res1: residue of first atom
    :param at1: first atom name
    :param res2: residue of second atom
    :param at2: second atom name
    """

    def __init__(self, res1, at1, res2, at2):
        self.res1_ = res1
        self.res2_ = res2
        self.at1_ = at1
        self.at2_ = at2
        super(NOEID, self).__init__("noe")

    def __hash__(self):
        return hash((self.res1_, self.at1_, self.res2_, self.at2_))

    def __eq__(self, other):
        if self.res1_ == other.res1_ and self.res2_ == other.res2_ \
                and self.at1_ == other.at1_ and self.at2_ == other.at2_:
            return True
        else:
            return False

    def __str__(self):
        s = "%i\t%s\t%i\t%s" % (self.res1_, self.at1_, self.res2_, self.at2_)
        return s

class PREID(BaseDataID):
    """
    ID objects for PRE back-calc values. PRE values are calculated between
    two atoms, so this requires residue of first atom, first atom name,
    residue of second atom and second atom name
    :param res1: residue of first atom
    :param at1: first atom name
    :param res2: residue of second atom
    :param at2: second atom name
    """

    def __init__(self, res1, at1, res2, at2):
        self.res1_ = res1
        self.res2_ = res2
        self.at1_ = at1
        self.at2_ = at2
        super(PREID, self).__init__("pre")

    def __hash__(self):
        return hash((self.res1_, self.at1_, self.res2_, self.at2_))

    def __eq__(self, other):
        if self.res1_ == other.res1_ and self.res2_ == other.res2_ \
                and self.at1_ == other.at1_ and self.at2_ == other.at2_:
            return True
        else:
            return False

    def __str__(self):
        s = "%i\t%s\t%i\t%s" % (self.res1_, self.at1_, self.res2_, self.at2_)
        return s

class rPREID(BaseDataID):
    """
    ID object for rPRE back-calc values. Requires residue and atom
    """

    def __init__(self, res_num, atom_name):
        self.res_ = res_num
        self.atom_ = atom_name
        super(rPREID, self).__init__("rpre")

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


class R2ID(BaseDataID):
    """
    ID object for R2 back-calc values. Requires residue and atom
    """

    def __init__(self, res_num, atom_name):
        self.res_ = res_num
        self.atom_ = atom_name
        super(R2ID, self).__init__("r2")

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


class Measurement(object):
    """
    Class for storing all of the info in a data measurement
    (experimental or structural). Contains a data ID and a value

    :param data_id: a BaseDataID object
    :param val: value of the measurement
    """

    def __init__(self, data_id=None, val=None):
        self.dataID_ = data_id
        self.val_ = val


class RunCRYSOL(object):
    """
    Class for running the SAXS back-calculator used by ENSEMBLE, CRYSOL.
    """

    def __init__(self):
        pass

    def run(self, pdb_path):
        """
        Run the program
        :param pdb_path: path of file to run predictor on
        :return: path to output file
        """
        # TODO implement this method
        return ""

    @staticmethod
    def read_output(path):
        """
        Read the output from the predictor
        :param path: path to output file
        :return: {RDCID: prediction} dict
        """
        # TODO implement this method
        return {SAXSID(0.00): 0.0}  # placeholder


class RunRDCCalculator(object):
    """
    Class for running the RDC back-calculator used by ENSEMBLE. I don't
    actually know how it is done, so this class is mostly empty, but
    has the structure we would need.
    """

    def __init__(self):
        pass

    def run(self, pdb_path):
        """
        Run the program
        :param pdb_path: path of file to run predictor on
        :return: path to output file
        """
        # TODO implement this method
        return ""

    @staticmethod
    def read_output(path):
        """
        Read the output from the predictor
        :param path: path to output file
        :return: {RDCID: prediction} dict
        """
        # TODO implement this method
        return {RDCID(0): 0.0}  # placeholder


class RunShiftX(object):
    """
    Class for running the SHIFTX2 command line program

    :param exe: path to SHIFTX2 executable
    :param tempdir: path to a directory to story temporary files
    """
    DEFAULT_EXE = "/usr/local/bin/shiftx2/shiftx2.py"
    DEFAULT_TEMPDIR = "./tmp/"

    def __init__(self, exe=DEFAULT_EXE, tempdir=DEFAULT_TEMPDIR):
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
    def read_ouput(path):
        """
        Reads a SHIFTX2 output file

        :param path: output file to be read
        :return: {ShiftID: shift_val} dict
        """
        try:
            f = open(path)
        except IOError:
            print "Could not find SHIFTX output file. It should have the same" \
                  "path as the pdb file, with '.cs' appended to the end." \
                  "Please name files correctly or use the 'RUN_SHIFTX' option" \
                  "in your input file."
            print "Aborting program"
            sys.exit()
        dout = {}
        i = 0

        for line in f:
            if i < 1:
                i += 1
                continue
            split = line.split(',')
            res_num = int(split[0])
            atom_name = split[2]
            shift_value = float(split[3])
            sid = ShiftID(res_num=res_num, atom_name=atom_name)
            dout[sid] = shift_value
            i += 1
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
    Read the J-coupling data for aB42 in ../test_data/

    :return: a {JCoupID: (val, err)} dict
    """
    jcoup_data_path = "../test/bax_2016_jcoup.txt"
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
    Read the shift data for aB42 in ../test_data/

    :return: a {ShiftID: val} dict
    """
    shift_data_path = "../test/filtered_ab42_cs.txt"
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


def read_jcoup_data(path):
    """
    Read a file of experimental j-coupling data. Must be in ENSEMBLE format:

    res1 atom1 res2 atom2 res3 atom3 res4 atom4 J err_low err_up

    Where the last lines are standard deviations.
    Currently the only usable type is 3JHNHA coupling data.
    :param path: path to data file
    :return: a {JCoupID: (val, err)} dict
    """
    jcoup_data = {}
    expj = open(path)
    for line in expj:
        split = line.split()
        if len(split) < 11 or line.startswith('#'):
            continue
        else:
            try:
                res_num = int(split[0])
                val = float(split[8])
                err = (float(split[9]) + float(split[10])) / 2
            except ValueError:
                print "J-coupling data not formatted properly. Please format " \
                      "as:\n res1 atom1 res2 atom2 res3 atom3 " \
                      "res4 atom4 J err_low err_up\n "
                print "Aborting program."
                sys.exit()

            jid = JCoupID(res_num)
            jcoup_data[jid] = (val, err)
    expj.close()
    return jcoup_data


def read_rdc_data(path):
    """
    Read a file of experimental RDC data. Must be in ENSEMBLE format:

    res1 atom1 res2 atom2 atom4 RDC

    Currently the only usable type intra-residue H-N data
    :param path: path to data file
    :return: a {RDCID: (val, err)} dict
    """
    rdc_data = {}
    expr = open(path)
    for line in expr:
        split = line.split()
        if len(split) < 5 or line.startswith('#'):
            continue
        else:
            try:
                res_num = int(split[0])
                val = float(split[4])
                err = float(split[5])
            except ValueError:
                print "RDC data not formatted properly. Please format " \
                      "as:\n res1 atom1 res2 atom2 RDC err"
                print "Aborting program."
                sys.exit()

            rid = JCoupID(res_num)
            rdc_data[rid] = (val, err)
    expr.close()
    return rdc_data


def read_saxs_data(path):
    """
    Read a file of experimental SAXS data. Must be in ENSEMBLE format:
    radius  intensity   error   Asym
    :param path: path to data file
    :return: {SAXSID: (val, err)} dictionary
    """
    saxs_data = {}
    exps = open(path)
    for line in exps:
        split = line.split()
        if len(split) < 3 or line.startswith('#'):
            continue
        else:
            try:
                rad = int(split[0])
                intensity = float(split[1])
                err = float(split[2])
            except ValueError:
                print "SAXS data not formatted properly. Please format " \
                      "as:\n radius  intensity   error   Asym"
                print "Aborting program."
                sys.exit()

            sid = SAXSID(rad)
            saxs_data[sid] = (intensity, err)
    exps.close()
    return saxs_data


def read_chemshift_data(path):
    """
    Read a file of experimental chemical shift data. Must be in ENSEMBLE format:

    residue atom shift error

    :param path: path to data file.
    :return: a {ShiftID: val} dict
    """
    shift_data = {}
    expcs = open(path)
    for line in expcs:
        split = line.split()
        if len(split) < 4 or line.startswith('#'):
            continue
        else:
            try:
                res_num = int(split[0])
                atom = str(split[1])
                val = float(split[2])
                err = float(split[3])
                # err = (float(split[9]) + float(split[10])) / 2
            except ValueError:
                print "Chemical shift data not formatted properly. Please " \
                      "format as:\n residue atom shift error"
                print "Aborting program."
                sys.exit()

            sid = ShiftID(res_num, atom)
            shift_data[sid] = (val, err)
    expcs.close()
    return shift_data


def get_md_energies():
    """
    Read the energy file containing all the energies for the
    MD ensemble in ../test_data/

    :return: {filename: energy} dict
    """
    path = "../test/MD_energies.txt"
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


def read_opt_out_file(path):
    """
    Read the output of an optimization run
    :param path: path to the optimization output file
    :return: list of pdb files of representing optimized ensemble
    """
    f = open(path)
    lines = f.readlines()
    f.close()
    files = []
    itr = int(lines[0].split()[-1])
    for line in lines[1:]:
        files.append(line.split()[0].strip('\n').strip(".cs"))
    return files, itr
