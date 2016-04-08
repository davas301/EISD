from MMTK.PDB import PDBConfiguration
from MMTK.Proteins import Protein

from readutil import RunShiftX, Measurement
from readutil import ShiftID

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
    """

    def __init__(self, pdbfile, shiftxfile=None, runshiftx=None, energy=None):
        """

        :param pdbfile: path to the pdbfile containing the representation of
        this structure
        :param shiftxfile: file containing back-calculated SHIFTX2 data for
        this structure
        :param runshiftx: an optional RunShiftX instance
        """
        self.pdb_ = pdbfile.split('/')[-1]  # only want the name of the pdb

        pdbconfig = PDBConfiguration(pdbfile)
        pdbconfig.deleteHydrogens()
        pepchains = pdbconfig.createPeptideChains()
        chain_idx = 0

        self.protein_ = Protein(pepchains[chain_idx])
        self.protein_.normalizeConfiguration()
        self.sequence_ = self.protein_.chains[0][0].sequence()

        if shiftxfile is not None:
            self.shiftxdata_ = RunShiftX.read_ouput(shiftxfile)
        elif runshiftx is not None:
            shiftxfile = runshiftx.run_shiftx_once(self.pdb_)
            self.shiftxdata_ = RunShiftX.read_ouput(shiftxfile)
        else:
            self.shiftxdata_ = None

        self.energy_ = energy
        self.dihed_ = self._get_all_dihed()
        self.coords_ = self._get_all_atom_coords()

    def _get_all_atom_coords(self):
        """
        Retrieve all of the atomic coordinates in this structure.
        :return: a {(res_num, atom_name): [x,y,z]} dict
        """
        dout = {}
        for atom in self.protein_.atomList():
            atom_name = atom.name
            res_num = atom.parent.parent.sequence_number
            idx = atom.index
            if atom.array is None:
                coords = atom.pos[0]
            else:
                coords = atom.array[idx]
            dout[(res_num, atom_name)] = coords
        return dout

    def _get_all_dihed(self):
        """
        Retrieve all phi, psi dihedral angles in this structure
        :return: a {res_num: (phi, psi)} dict
        """
        all_res = self.protein_.residues()
        all_dihed = {}
        for i in range(0, len(all_res)):
            phi_psi = all_res[i].phiPsi()
            phi = phi_psi[0]
            psi = phi_psi[1]
            res_num = all_res[i].sequence_number
            all_dihed[res_num] = (phi, psi)
        return all_dihed

    def get_struct_measure(self, exp_id):
        """
        Get a structural measurement from a DataID corresponding to an
        experimental measurement. This structural measurement can be
        input into a back-calculator
        :param exp_id: An experimental data ID (ShiftID or JCoupID)
        """
        if isinstance(exp_id, ShiftID):
            struct_meas = Measurement(data_id=exp_id,
                                      val=self.shiftxdata_[exp_id])
        else:
            struct_meas = Measurement(data_id=exp_id,
                                      val=self.dihed_[exp_id.res_])
        return struct_meas
