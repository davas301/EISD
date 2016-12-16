import Bio.PDB
from readutil import RunShiftX, Measurement, RunRDCCalculator, RunCRYSOL
from readutil import ShiftID, RDCID, RHID, SAXSID

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
Module for storing and manipulating protein structures and ensembles
"""


class Structure(object):
    """
    Class for representing and modifying protein structures. Interfaces
    significantly with MMTK.

    :param pdbfile: path to the pdbfile containing the representation of this structure
    :param shiftxfile: file containing back-calculated SHIFTX2 data for this structure
    :param runshiftx: an optional RunShiftX instance
    :param rdcfile: otpional file containing back-calculated RDC data for this structure
    :param runrdccalc: an optional RunRDCCalculator instance
    :param rh_val: an optional back-calculated tuple for hydrodynamic radius and error
    :param saxsfile: otpional file containing back-calculated SAXS data for this structure
    :param run_crysol: an option RunCRYSOL instance
    """

    def __init__(self, pdbfile, shiftxfile=None, runshiftx=None,
                 rdcfile=None, runrdccalc=None, rh_val=None,
                 saxsfile=None, run_crysol=None):
        self.pdb_ = pdbfile.split('/')[-1]  # only want the name of the pdb

        parser = Bio.PDB.PDBParser()
        struct = parser.get_structure(self.pdb_, pdbfile)
        chain = struct.get_chains().next()
        self.protein_ = Bio.PDB.Polypeptide.Polypeptide(chain)

        # check for SHIFTX data
        if shiftxfile is not None:
            self.shiftxdata_ = RunShiftX.read_ouput(shiftxfile)
        elif runshiftx is not None:
            shiftxfile = runshiftx.run_shiftx_once(self.pdb_)
            self.shiftxdata_ = RunShiftX.read_ouput(shiftxfile)
        else:
            self.shiftxdata_ = None

        # check for RDC back-calculated data:
        if rdcfile is not None:
            self.rdcData_ = RunRDCCalculator.read_output(rdcfile)
        elif runrdccalc is not None:
            rdcfile = runrdccalc.run(self.pdb_)
            self.rdcData_ = RunRDCCalculator.read_output(rdcfile)
        else:
            self.rdcData_ = None

        # check for RDC back-calculated data:
        if saxsfile is not None:
            self.saxsData_ = RunCRYSOL.read_output(saxsfile)
        elif run_crysol is not None:
            saxsfile = run_crysol.run(self.pdb_)
            self.saxsData_ = RunCRYSOL.read_output(saxsfile)
        else:
            self.saxsData_ = None

        # check if there is a back-calculated RH value
        if rh_val is not None:
            self.rhVal_ = rh_val

        self.dihed_ = self._get_all_dihed()

    def _get_all_dihed(self):
        """
        Retrieve all phi, psi dihedral angles in this structure

        :return: a {res_num: (phi, psi)} dict
        """
        phi_psi = self.protein_.get_phi_psi_list()
        all_dihed = {}
        for i in range(0, len(phi_psi)):
            all_dihed[i + 1] = phi_psi[i]
        return all_dihed

    def get_struct_measure(self, exp_id):
        """
        Get a structural measurement from a DataID corresponding to an
        experimental measurement. This structural measurement can be
        input into a back-calculator

        :param exp_id: An experimental data ID
        """
        if exp_id.dtype_ == "shift":
            struct_meas = Measurement(data_id=exp_id,
                                      val=self.shiftxdata_[exp_id])
        elif exp_id.dtype_ == "rdc":
            struct_meas = Measurement(data_id=exp_id,
                                      val=self.rdcData_[exp_id])
        elif exp_id.dtype_ == "rh":
            struct_meas = Measurement(data_id=exp_id,
                                      val=self.rhVal_)
        elif exp_id.dtype_ == "saxs":
            struct_meas = Measurement(data_id=exp_id,
                                      val=self.saxsData_[exp_id])
        else:  # dtype==jcoup
            struct_meas = Measurement(data_id=exp_id,
                                      val=self.dihed_[exp_id.res_])
        return struct_meas
