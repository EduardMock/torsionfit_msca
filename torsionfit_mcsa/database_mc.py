import numpy as np
import pandas as pd

import simtk.openmm as mm
from simtk.unit import *
# from simtk.openmm.app import *   # not compatible

# from simtk.unit import Quantity, nanometers, angstroms, kilojoules_per_mole, kilocalorie_per_mole, picoseconds
from  mdtraj import Trajectory

from . import parameters as par

import matplotlib.pyplot as plt
import ipywidgets 
import IPython.display as display
from ipywidgets import Output, Tab, GridspecLayout, Layout, ButtonStyle
from IPython.display import clear_output

from .utils import *
from parmed.exceptions import ParameterError
from parmed.topologyobjects import (Dihedral,  DihedralType, ImproperType,
                                ExtraPoint, DrudeAtom)

from copy import deepcopy
from copy import copy as _copy
from fnmatch import fnmatch



class DataBase(Trajectory):
    """container object for molecular configurations and energies.

    Attributes
    ----------
    structure: ParmEd.Structure
    mm_energy: simtk.unit.Quantity((n_frames), unit=kilojoule/mole)
    positions:
    context:
    system:
    integrator:
    """

    def __init__(self, pos, topology, structure=None, time=None, gro=False):
        """Create new TorsionScanSet object"""
        assert isinstance(topology, object)
        super(DataBase, self).__init__(pos, topology, time)
        
        self.gro = gro
        self.structure = structure
        
        # if self.gro:
        #     self.structure = topology    
        
        self.positions = pos
        self.mm_energy = Quantity()
        self.context = None
        self.system = None
        self.dihedral_opt= None
        
        # mm.VerletIntegrator(0.004*picoseconds)

        # Don't allow an empty TorsionScanSet to be created
        if self.n_frames == 0:
            msg = 'DataBase has no frames!\n'
            msg += '\n'
            msg += 'DataBase provided were:\n'
            msg += str(positions)
            raise Exception(msg)
        
        

    def create_context(self, param=None, platform=None):
        """

        Parameters
        ----------
        param :
        platform :
        """
        if self.gro is False:
            self.structure.load_parameters(param)
            self.system = self.structure.createSystem(param,nonbondedMethod= mm.app.NoCutoff, constraints=None)
            
        self.system = self.structure.createSystem(nonbondedMethod= mm.app.NoCutoff, constraints=None)
        self.integrator=mm.VerletIntegrator(0.004*picoseconds)
        if platform != None:
            self.context = mm.Context(self.system, self.integrator, platform)
        else:
            self.context = mm.Context(self.system, self.integrator)
 



    def copy_torsions(self, param, platform=None):
        """

        Parameters
        ----------
        param :
        platform :
        """
        forces = {self.system.getForce(i).__class__.__name__: self.system.getForce(i)
                  for i in range(self.system.getNumForces())}
        torsion_force = forces['PeriodicTorsionForce']

        # create new force
        new_torsion_force = self.structure.omm_dihedral_force()

        # sanity check
        if torsion_force.getNumTorsions() != new_torsion_force.getNumTorsions():
            # create new context and new integrator. First delete old context and integrator
            del self.system
            del self.context
            del self.integrator
            self.integrator = mm.VerletIntegrator(0.004*picoseconds)
            self.create_context(param, platform)
            forces = {self.system.getForce(i).__class__.__name__: self.system.getForce(i)
                      for i in range(self.system.getNumForces())}
            torsion_force = forces['PeriodicTorsionForce']

        # copy parameters
        for i in range(new_torsion_force.getNumTorsions()):
            torsion = new_torsion_force.getTorsionParameters(i)
            torsion_force.setTorsionParameters(i, *torsion)
        # update parameters in context
        torsion_force.updateParametersInContext(self.context)

        # clean up
        del new_torsion_force



    def _string_summary_basic(self):
        """Basic summary of TorsionScanSet in string form."""
        energy_str = 'with MM Energy' if self._have_mm_energy else 'without MM Energy'
        value = "Database with %d frames, %d atoms, %d residues, %s" % (
                     self.n_frames, self.n_atoms, self.n_residues, energy_str)
        return value


    
    

    def compute_energy(self, param=None, platform=None,justcomp=True):
        """ Computes energy for a given structure with a given parameter set

        Parameters
        ----------
        offset :
        param: parmed.charmm.CharmmParameterSet
        platform: simtk.openmm.Platform to evaluate energy on (if None, will select automatically)
        """

        if self.n_frames == 0:
            raise Exception("self.n_frames = 0! There are no frames to compute energy for.")

        # Check if context exists.
        if not self.context:
            self.create_context(param, platform)
        else:
            # copy new torsion parameters
            self.copy_torsions(param, platform)

        # Compute potential energies for all snapshots.
        
        self.mm_energy = Quantity(value=np.zeros([self.n_frames], np.float64),unit=kilojoule_per_mole)
        
        if justcomp:
            for i in range(self.n_frames):
                self.context.setPositions(self.positions[i])
                state = self.context.getState(getEnergy=True)
                self.mm_energy[i] = state.getPotentialEnergy()
        else:
            
            integrator = mm.LangevinIntegrator(
                        300*kelvin,       # Temperature of heat bath
                        1.0/picoseconds,  # Friction coefficient
                        2.0*femtoseconds # Time step
                        )
            
            
            for i in range(self.n_frames):
                
                simulation = app.Simulation(self.structure, self.system, integrator, platform)
                sys=simulation.context.setPositions(self.positions[i])
                sys_f=freeze_torsions(sys,self.torsion_index[i],self.angle[i])
                sys_f.minimizeEnergy()
                state = sys_f.context.getState(getEnergy=True,getPositions=True)
                self.mm_energy[i] = state.getPotentialEnergy()




    def freeze_torsions(system, torsions_to_freeze, torsions_angles):
        
        energy_expression = f'-fc*cos(theta-theta0)'
        # fc = unit.Quantity(k, unit.kilojoule_per_mole)
        restraint = CustomTorsionForce(energy_expression)
        # restraint.addGlobalParameter('fc', fc)
        restraint.addPerTorsionParameter('theta0')

        for torsion, angle in zip(torsions_to_freeze, torsions_angles):
            torsion_id = restraint.addTorsion(*torsion)
            restraint.setTorsionParameters(torsion_id, *torsion, [angle * np.pi / 180.0])

        system.addForce(restraint)

        return system


                
    
    # def show_torsions(self, xyz, di_num, di_name= None, increment=30, ):
        
    #     molecule=Molecule(xyz)
        
    #     row= int(len(di_num)/4)
    #     if len(di_num)%4 > 0:
    #         row +=1
            
    #     grid = GridspecLayout(row, 4)

        
    #     if di_name is None:
    #         atomtypes=[i.type for i in self.structure.atoms]    
    #         di_name=["%s_%s_%s_%s"%(atomtypes[i[0]], atomtypes[i[1]],atomtypes[i[2]],atomtypes[i[3]]) for i in di_num]
    #         # conv.append([ [atomtypes[i[0]], atomtypes[i[1]],atomtypes[i[2]],atomtypes[i[3]]] for i in di_num])

    #     di_con=[]
    #     c=0
    #     r=0
    #     o=0
    #     for d in di_num:
    #         m_all, goAtoms = molecule.rotate_bond(0,*d[1:3],increment=increment)
    #         m_rot=m_all[::]
    #         molview = nglview.NGLWidget(MyStructureTrajectory(m_rot)) #._set_size(200,200)
    #         molview._set_size('500px', '500px')
    #         text = ipywidgets.Button(description= "%s"%di_name[o],
    #                     layout=Layout(width='500px', grid_area='header'),
    #                     style=ButtonStyle(button_color='lightblue',font_size="35",font_weight='bold'))
    #         grid[r,c]= ipywidgets.VBox([text,molview])
    #         c+=1
    #         o+=1
    #         if c==4:
    #             r+=1
    #             c=0
        
        
    #     return display.display(grid)




    def compute_dipoles(self, param, platform=None):
        
            if self.n_frames == 0:
                raise Exception("self.n_frames = 0! There are no frames to compute energy for.")

            if not self.context:
                self.create_context(param, platform)
            else:
                self.copy_torsions(param, platform)

            # Compute potential energies for all snapshots.
            self.mm_dipole = np.zeros([len(self.positions),4]) #Quantity(value=np.zeros([len(self.positions)], np.float64), unit=debye)
            for i in range(len(self.positions)):
                self.context.setPositions(self.positions[i])
                self.mm_dipole[i] = self.dipole()
                

    def fcog(self):  # Center of geometry
        posi = self.context.getState(getPositions=True).getPositions()
        posi = posi.value_in_unit(angstroms)
        cogx = 0.0; cogy= 0.0; cogz=0.0
        for i in range(len(posi)):    
            cogx = cogx + posi[i][0]
            cogy = cogy + posi[i][1]
            cogz = cogz + posi[i][2]
        cogx = cogx/len(posi); cogy = cogy/len(posi); cogz = cogz/len(posi) 
        cog = (cogx,cogy,cogz)
        # print("Center of Geometry found at: ", cog)
        return(cog)


    def fcom(self):   # Center of mass
        posi = self.context.getState(getPositions=True).getPositions()
        posi = posi.value_in_unit(angstroms)

        # Center of geometry, will be moved to a separate function
        comx = 0.0; comy= 0.0; comz=0.0
        mass = []
        for i in range(len(posi)):   
            mass.append(self.system.getParticleMass(i))
            comx = comx + posi[i][0]*mass[i]._value
            comy = comy + posi[i][1]*mass[i]._value
            comz = comz + posi[i][2]*mass[i]._value
        totmass = sum(mass)
        comx = comx/totmass._value; comy = comy/totmass._value; comz = comz/totmass._value 
        com = (comx,comy,comz)
        # print("Center of Mass found at: ", com)
        return (com)



    def dipole(self,qcog=False,qcom=False,qxyz=True,dunit=4.8032412710,fwrite=False):
        
        pos = self.context.getState(getPositions=True).getPositions()
        # print(pos)
        pos = pos.value_in_unit(angstroms)
        cog = self.fcog()
        
        if qcom:
            com = self.fcom(simulation)
        
        forces = { force.__class__.__name__ : force for force in self.system.getForces() }
        reference_force = forces['NonbondedForce']
        charges=[]
        for index in range(reference_force.getNumParticles()):
            [charge, sigma, epsilon] = reference_force.getParticleParameters(index)
            charges.append(charge)
        # print("Sum of charges =", sum(charges) )
        
        
        dipocx =[]    
        dipocy =[]    
        dipocz =[]    
        for i in range(len(pos)):    
            if qxyz:
                xd=pos[i][0] 
                yd=pos[i][1] 
                zd=pos[i][2] 
                dipocx.append(xd*charges[i])
                dipocy.append(yd*charges[i])
                dipocz.append(zd*charges[i])
            if qcog:
                xd=pos[i][0] - cog[0]
                yd=pos[i][1] - cog[1]
                zd=pos[i][2] - cog[2]
                dipocx.append(xd*charges[i])
                dipocy.append(yd*charges[i])
                dipocz.append(zd*charges[i])
            if qcom:   
                xd=pos[i][0] - com[0]
                yd=pos[i][1] - com[1]
                zd=pos[i][2] - com[2]
                dipocx.append(xd*charges[i])
                dipocy.append(yd*charges[i])
                dipocz.append(zd*charges[i])
        dipx = sum(dipocx)._value*dunit
        dipy = sum(dipocy)._value*dunit
        dipz = sum(dipocz)._value*dunit
        diptot = math.sqrt(dipx**2+dipy**2+dipz**2)

        return [dipx,dipy,dipz,diptot]


    @property
    def _have_mm_energy(self):
        return len(self.mm_energy) != 0

    def __getitem__(self, key):
        "Get a slice of this trajectory"
        return self.slice(key)


    def load_parameters_gro(self, parmset, copy_parameters=True):
        """
        Loads parameters from a parameter set.
        Parameters
        ----------
        parmset : :class:`struture.parameterset`
            List of all parameters
        copy_parameters : bool, optional, default=True
            If False, parmset will not be copied.
            WARNING:
            -------
            Not copying parmset will cause ParameterSet and Structure to share
            references to types.  If you modify the original parameter set, the
            references in Structure list_types will be silently modified.
            However, if you change any reference in the parameter set, then that
            reference will no longer be shared with structure.
            Example where the reference in ParameterSet is changed. The
            following will NOT modify the parameters in the psf::
                psf.load_parameters(parmset, copy_parameters=False)
                parmset.angle_types[('a1', 'a2', a3')] = AngleType(1, 2)
            The following WILL change the parameter in the psf because the
            reference has not been changed in ``ParameterSet``::
                psf.load_parameters(parmset, copy_parameters=False)
                a = parmset.angle_types[('a1', 'a2', 'a3')]
                a.k = 10
                a.theteq = 100
            Extra care should be taken when trying this with dihedral_types.
            Since dihedral_type is a Fourier sequence, ParameterSet stores
            DihedralType for every term in DihedralTypeList. Therefore, the
            example below will STILL modify the type in the :class:`Structure`
            list_types::
                parmset.dihedral_types[('a', 'b', 'c', 'd')][0] = DihedralType(1, 2, 3)
            This assigns a new instance of DihedralType to an existing
            DihedralTypeList that ParameterSet and Structure are tracking and
            the shared reference is NOT changed.
            Use with caution!
        Notes
        -----
        - If any dihedral or improper parameters cannot be found, I will try
          inserting wildcards (at either end for dihedrals and as the two
          central atoms in impropers) and see if that matches.  Wild-cards will
          apply ONLY if specific parameters cannot be found.
        - This method will expand the dihedrals attribute by adding a separate
          Dihedral object for each term for types that have a multi-term
          expansion
        Raises
        ------
        ParameterError if any parameters cannot be found
        """
        if copy_parameters:
            parmset = _copy(parmset)
        self.structure.combining_rule = parmset.combining_rule
        
        # First load the atom types
        for atom in self.structure.atoms:
            try:
                if isinstance(atom.type, int):
                    atype = parmset.atom_types_int[atom.type]
                else:
                    atype = parmset.atom_types_str[atom.type]
            except KeyError:
                raise ParameterError(f"Could not find atom type for {atom.type}")
            atom.atom_type = atype
            # Change to string type to look up the rest of the parameters. Use
            # upper-case since all parameter sets were read in as upper-case
            atom.type = str(atom.atom_type)
            atom.atomic_number = atype.atomic_number


        # Next load all of the bonds
        skipped_bonds = set()
        for bond in self.structure.bonds:
            # Skip any bonds with drude atoms or virtual sites. They are not stored.
            # Depending on how Drude support is implemented in Amber (if that ever happens), we
            # may have to add dummy values here.
            if isinstance(bond.atom1, (DrudeAtom, ExtraPoint)) or isinstance(bond.atom2, (DrudeAtom, ExtraPoint)):
                skipped_bonds.add(bond)
                continue
            # Construct the key
            key = (min(bond.atom1.type, bond.atom2.type), max(bond.atom1.type, bond.atom2.type))
            try:
                bond.type = parmset.bond_types[key]
            except KeyError:
                raise ParameterError(f"Missing bond type for {bond}")
            bond.type.used = False
        # Build the bond_types list
        del self.structure.bond_types[:]
        
        for bond in self.structure.bonds:
            if bond in skipped_bonds:
                continue
            if bond.type.used:
                continue
            bond.type.used = True
            self.structure.bond_types.append(bond.type)
            bond.type.list = self.structure.bond_types
            
            
        # Next load all of the angles. If a Urey-Bradley term is defined for
        # this angle, also build the urey_bradley and urey_bradley_type lists
        # for ang in self.structure.angles:
        #     # Construct the key
        #     key = (min(ang.atom1.type, ang.atom3.type),
        #            ang.atom2.type,
        #            max(ang.atom1.type, ang.atom3.type))
        #     try:
        #         ang.type = parmset.angle_types[key]
        #         ang.type.used = False
        #         ubt = parmset.urey_bradley_types[key]
        #         if ubt is not NoUreyBradley:
        #             ub = UreyBradley(ang.atom1, ang.atom3, ubt)
        #             self.urey_bradleys.append(ub)
        #             ubt.used = False
        #     except KeyError:
        #         raise ParameterError(f"Missing angle type for {ang}")
            
        del self.structure.angle_types[:]
        for ang in self.structure.angles:
            # print(ang)
            # if ang.type.used:
            #     continue
            ang.type.used = True
            self.structure.angle_types.append(ang.type)
            ang.type.list = self.structure.angle_types
            
        # Next load all of the dihedrals.
        active_dih_list = set()
        for dih in self.structure.dihedrals:
            # Store the atoms
            a1, a2, a3, a4 = dih.atom1, dih.atom2, dih.atom3, dih.atom4
            key = (a1.type, a2.type, a3.type, a4.type)
            # First see if the exact dihedral is specified
            if not key in parmset.dihedral_types:
                # Check for wild-cards
                key = ('X', a2.type, a3.type, 'X')
                if not key in parmset.dihedral_types:
                    raise ParameterError(f'No dihedral parameters found for {dih}')
            dih.type = parmset.dihedral_types[key]
            dih.type.used = False
            pair = (dih.atom1.idx, dih.atom4.idx) # To determine exclusions
            if (dih.atom1 in dih.atom4.bond_partners or
                dih.atom1 in dih.atom4.angle_partners):
                dih.ignore_end = True
            elif pair in active_dih_list:
                dih.ignore_end = True
            else:
                active_dih_list.add(pair)
                active_dih_list.add((dih.atom4.idx, dih.atom1.idx))
        del self.structure.dihedral_types[:]
        for dihedral in self.structure.dihedrals:
            if dihedral.type.used:
                continue
            dihedral.type.used = True
            self.structure.dihedral_types.append(dihedral.type)
            dihedral.type.list = self.structure.dihedral_types
            
        # Now do the impropers
        for imp in self.structure.impropers:
            a1, a2, a3, a4 = imp.atom1.type, imp.atom2.type, imp.atom3.type, imp.atom4.type
            imp.type = parmset.match_improper_type(a1, a2, a3, a4)
            if imp.type is None:
                raise ParameterError(f"No improper type for {a1}, {a2}, {a3}, and {a4}")
            imp.type.used = False
        # prepare list of harmonic impropers present in system
        del self.structure.improper_types[:]
        for improper in self.structure.impropers:
            if improper.type.used:
                continue
            improper.type.used = True
            if isinstance(improper.type, ImproperType):
                self.structure.improper_types.append(improper.type)
                improper.type.list = self.structure.improper_types
            elif isinstance(improper.type, DihedralType):
                self.structure.dihedral_types.append(improper.type)
                improper.type.list = self.structure.dihedral_types
            else:
                assert False, 'Should not be here'
        # Look through the list of impropers -- if there are any periodic
        # impropers, move them over to the dihedrals list
        for i in reversed(range(len(self.structure.impropers))):
            if isinstance(self.structure.impropers[i].type, DihedralType):
                imp = self.structure.impropers.pop(i)
                dih = Dihedral(imp.atom1, imp.atom2, imp.atom3, imp.atom4,
                               improper=True, ignore_end=True, type=imp.type)
                imp.delete()
                self.structure.dihedrals.append(dih)
                
                

