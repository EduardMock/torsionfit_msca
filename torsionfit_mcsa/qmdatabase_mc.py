import pandas as pd
import numpy as np

from simtk.unit import Quantity, nanometers, kilojoules_per_mole, kilocalories_per_mole, picoseconds, angstroms, kelvin,femtoseconds
from simtk.openmm import app
import simtk.openmm as mm


from cclib.parser import gaussianparser, psi4parser
from cclib.parser.utils import convertor

import mdtraj as md
from parmed.charmm import CharmmPsfFile, CharmmParameterSet
import parmed
# from torsionfit.database import DataBase
from .database_mc import DataBase

from copy import deepcopy
from fnmatch import fnmatch
import os

import matplotlib.pyplot as plt
import ipywidgets 
import IPython.display as display
from ipywidgets import Output, Tab, GridspecLayout, Layout, ButtonStyle
from IPython.display import clear_output

from .utils import *


from copy import deepcopy
from fnmatch import fnmatch
import os
import re
import warnings
import itertools
import seaborn as sns



def parse_psi4_out(oufiles_dir,xyz, structure, pattern="*.out", gro= False):
    """
    Parse psi4 out files from distributed torsion scan (there are many output files, one for each structure)
    :param oufiles_dir: str
        path to directory where the psi4 output files are
    :param structure: str
        path to psf, mol2 or pbd file of structure
    :param pattern: str
        pattern for psi4 output file. Default is *.out
    :return: TorsionScanSet

    """
    
    molecule= Molecule(xyz)
    
    # Check extension of structure file
    if gro:
        # topology = md.load(structure).topology
        structure = parmed.load_file(structure)
        topology = structure.topology
        
    else:
        topology = md.load_psf(structure)
        structure=parmed.load_file(structure)
        # structure = CharmmPsfFile(structure)


    torsion_angles=[i for i in os.listdir(oufiles_dir)]
    print("This torsions habe been found:")
    print(torsion_angles)
    
    out_files = {}
    for ta in torsion_angles:
        for path, subdir, files in os.walk("%s/%s/opt_tmp/"%(oufiles_dir,ta)):
            # print(files)
            for name in files:
                if fnmatch(name, pattern):
                    if name.startswith('timer'):
                        continue
                    
                    try:
                        out_files[ta]
                    except KeyError:
                        out_files[ta] = []
                    path = os.path.join(os.getcwd(), path, name)
                    out_files[ta].append(path)         

    
    #extract only one torsion
    
    # out_files = {}
    # for path, subdir, files in os.walk(oufiles_dir):
    #     # print(files)
    #     for name in files:
    #         if fnmatch(name, pattern):
    #             if name.startswith('timer'):
    #                 continue
    #             # name_split = name.split('_')
    #             name_split = oufiles_dir.split('/')
    #             try:
    #                 # torsion_angle = (name_split[1] + '_' + name_split[2] + '_' + name_split[3] + '_' + name_split[4])
    #                 torsion_angle = (name_split[-2])

    #             except IndexError:
    #                 warnings.warn("Do you only have one torsion scan? The output files will be treated as one scan")
    #                 torsion_angle = 'only_one_scan'
    #             try:
    #                 out_files[torsion_angle]
    #             except KeyError:
    #                 out_files[torsion_angle] = []
    #             path = os.path.join(os.getcwd(), path, name)
    #             out_files[torsion_angle].append(path)
    # print(out_files)
    
    
    
    sorted_files = []
    dih_angles = []
    for tor in out_files:
        
        dih_angle = []
        for file in out_files[tor]:
            dih_angle.append(int(file.split('_')[-1].split('/')[0]))
        sorted_files.append([out_file for (angle, out_file) in sorted(zip(dih_angle, out_files[tor]))])
        dih_angle.sort()
        dih_angles.append(dih_angle)
    if not out_files:
        raise Exception("There are no psi4 output files. Did you choose the right directory?")

    # print(sorted_files)

    if gro:
        positions = np.ndarray((0, topology.getNumAtoms(), 3))
    else:
        positions = np.ndarray((0, topology.n_atoms, 3))
    qm_energies = np.ndarray(0)
    torsions = np.ndarray((0, 4), dtype=int)
    angles = np.ndarray(0, dtype=float)
    optimized = np.ndarray(0, dtype=bool)



    # Parse files
    bad_keys=[]
    bk=0
    for f in itertools.chain.from_iterable(sorted_files):
        
        
        #collect torsions
        # torsion = np.ndarray((1, 4), dtype=int)
        path=f.split('scan.log')
        dihedral_num=np.loadtxt("%s/dihedrals.txt"%(path[0]), skiprows=2, dtype=int )
        torsions = np.append(torsions, [[ x-1 for x in dihedral_num]], axis=0)
        # print(torsions)
        
        
        #collect optimized structures and coords
        optimizer = True
        log = psi4parser.Psi4(f)
        data = log.parse()
        try:
            data.optdone
        except AttributeError:
            optimizer = False
            warnings.warn("Warning: Optimizer failed for {}".format(f))
        optimized = np.append(optimized, [optimizer])
        positions = np.append(positions, data.atomcoords[-1][np.newaxis]*0.1, axis=0)
        positions = Quantity(value=positions, unit=nanometers) #.in_units_of(angstroms)
        
        
        
        # Try to collect MP2 energies. Otherwise take SCFenergies
        try:
            qm_energy = convertor(data.mpenergies[-1], "eV", "kJ/mol")
            # print(qm_energy)
        except AttributeError:
            try:
                qm_energy = convertor(np.array([data.scfenergies[-1]]), "eV", "kJ/mol")
                bad_keys.append(bk)
            except AttributeError:
                warnings.warn("Warning: Check if the file terminated before completing SCF")
                qm_energy = np.array([np.nan])
                
        qm_energies = np.append(qm_energies, qm_energy, axis=0)
        qm_energies=Quantity(value=qm_energies, unit=kilojoules_per_mole)
        bk+=1


    # Subtract lowest energy to find relative energies
    # qm_energies = qm_energies - min(qm_energies)
    
    angles = np.asarray(list(itertools.chain.from_iterable(dih_angles)))
    
    # if len(bad_keys) > 0:
    #     for bk in bad_keys:
    #         positions=np.delete(positions,bk,axis=0)
    #         angles=np.delete(angles,bk,axis=0)
    #         qm_energies =np.delete(qm_energies,bk,axis=0)
    #         optimized =np.delete(optimized,bk,axis=0)
    #         torsions=np.delete(torsions,bk,axis=0)
    
    
    d_name=torsions.copy()
    _, idx, counts =np.unique(d_name,return_index=True,return_counts=True, axis=0)
    d_name_unique=d_name[np.sort(idx)]
    # d_name_unique=d_name[idx]
    extracted_torsions=dict(zip(torsion_angles, zip(d_name_unique.tolist(),counts) ))
    
    
    # _, idx, counts =np.unique(d_name,return_index=True,return_counts=True, axis=0)
    # # d_name_unique=d_name[np.sort(idx)]
    # d_name_unique=d_name[idx]
    # extracted_torsions=dict(zip(torsion_angles, d_name_unique.tolist() ))
    
    
    # print(qm_energies)
    return QMDataBase(molecule=molecule, positions=positions, topology=topology, structure=structure, angles=angles,
                      qm_energies=qm_energies, optimized=optimized, torsion_index=torsions, extracted_torsions=extracted_torsions, gro=gro) 
        
    

# def count_unique(torsion_index):
#     d_name=torsion_index.copy()
#     _, idx, counts =np.unique(d_name,return_index=True,return_counts=True, axis=0)
#     return counts



class QMDataBase(DataBase):
    """container object for torsion scan

    A TorsionScanSet should be constructed by loading Gaussian 09 torsion scan log files or a psi4 output file from disk
    with an mdtraj.Topology object

    Attributes
    ----------
    structure: ParmEd.Structure
    qm_energy: simtk.unit.Quantity((n_frames), unit=kilojoule/mole)
    mm_energy: simtk.unit.Quantity((n_frames), unit=kilojoule/mole)
    torsion_index: {np.ndarray, shape(n_frames, 4)}
    """

    def __init__(self,molecule, positions, topology, structure,  qm_energies, mm_energies=Quantity(), delta_energies=Quantity(),
                 phis=None, angles=None, optimized=None, torsion_index=None, mm_opt_positions=None,extracted_torsions=None,time=None,
                 gro=False, energy_unit=kilojoules_per_mole):
        """Create new TorsionScanSet object"""
        assert isinstance(topology, object)
        super(QMDataBase, self).__init__(positions, topology, structure, time, gro)
        self._energy_unit = energy_unit
        
        self.molecule=molecule
        self.qm_energy = qm_energies
        self.mm_energy = mm_energies
        self.mm_opt_positions = mm_opt_positions
        self.mm_energy_min=None
        self.delta_energy = delta_energies
        self.initial_mm = Quantity()
        
        self.d_name= extracted_torsions
        self.torsion_index = torsion_index
        self.angles = angles
        self.optimized = optimized
        self.phis = phis
        self.mm_dipole=None

        

        

    @property
    def energy_unit(self):
        return self._energy_unit   

    @energy_unit.setter
    def energy_unit(self, unit):
        """Change energy to choosen unit"""
        self._energy_unit = unit 
        self.qm_energy = self.qm_energy.in_units_of(unit)
        self.mm_energy = self.mm_energy.in_units_of(unit)


    def update_d_name(self,torsion_index):
        t_index= torsion_index.copy()
        dn= list(self.d_name.keys())
        unique, idx, counts =np.unique(t_index,return_index=True,return_counts=True, axis=0)
        # counts=[counts[id] for id in sorted(idx)]
        t_index_unique=t_index[np.sort(idx)]
        
        c_sorted=[]
        for t in t_index_unique:
            for u,c in zip(unique ,counts):
                if np.array_equal(t,u):
                    c_sorted.append(c)
        d_name=dict( zip(dn, zip(t_index_unique.tolist(),c_sorted )))
        
        return d_name

    def make_reference(self):
        self.qm_energy-= self.qm_energy.min()
        self.mm_energy-= self.mm_energy.min()
    
    # def compute_energy(self, param=None, offset=None, platform=None):
    #     """ Computes energy for a given structure with a given parameter set

    #     Parameters
    #     ----------
    #     param: parmed.charmm.CharmmParameterSet
    #     platform: simtk.openmm.Platform to evaluate energy on (if None, will select automatically)
    #     """

    #     # Save initial mm energy
    #     save = False
    #     if not self._have_mm_energy:
    #         save = True

    #     # calculate energy
    #     super(QMDataBase, self).compute_energy(param, platform)

    #     self.energy_unit = self.energy_unit
    #     # min_energy = self.mm_energy.min()
    #     # self.mm_energy -= min_energy
        
    #     if save:
    #         self.initial_mm = deepcopy(self.mm_energy)
    #     if offset:
    #         offset = Quantity(value=offset.value, unit=self.mm_energy.unit)
    #         self.mm_energy += offset
        
    #     # self.delta_energy = (self.qm_energy - self.mm_energy)
    #     # self.delta_energy = self.delta_energy - self.delta_energy.min()
    
    # @staticmethod
    # def f_torsion(system, torsions_to_freeze, torsions_angles):
    #     # print()
    #     energy_expression = f'cos(theta-theta0)'
    #     # fc = unit.Quantity(0.133, unit.kilojoule_per_mole)
    #     restraint = mm.CustomTorsionForce(energy_expression)
    #     # restraint.addGlobalParameter('fc', fc)
    #     restraint.addPerTorsionParameter('theta0')

    #     # for torsion, angle in zip(torsions_to_freeze, torsions_angles):
    #     torsion_id = restraint.addTorsion( *torsions_to_freeze )
    #     # print(torsions_to_freeze[0],torsions_to_freeze[1],torsions_to_freeze[2],torsions_to_freeze[3])
    #     restraint.setTorsionParameters(torsion_id, *torsions_to_freeze, [torsions_angles * np.pi / 180.0])

    #     # system.addForce(restraint)
    #     return system
    
    
    @staticmethod
    def freeze_atoms(system, atom_list):
        for at in atom_list:
            system.setParticleMass(int(at), 0)
        return system
    
    # def harmonic_restraint(system, atom_list):
    #     restraint = app.forces.HarmonicRestraintForce(spring_constant=1000 * unit.kilocalories_per_mole / unit.angstrom**2,
    #                                                   restrained_atom_indices1=ligand_atom_list,
    #                                                   restrained_atom_indices2=receptor_atom_list)
    #     system.addForce(restraint)
        
    #     return system
    
    def create_restraint(self,restrain_k):
        self.restraint_frc_index = None
        self.restrain_k=restrain_k
        if self.restrain_k != 0.0:
            restraint_frc = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
            restraint_frc.addGlobalParameter("k", self.restrain_k * kilocalories_per_mole / angstroms**2)
            restraint_frc.addPerParticleParameter("x0")
            restraint_frc.addPerParticleParameter("y0")
            restraint_frc.addPerParticleParameter("z0")
            for i,j in enumerate(np.arange(len(self.molecule.elem))):
                restraint_frc.addParticle(int(j))
                restraint_frc.setParticleParameters(int(i),int(j), [0.0, 0.0, 0.0])
            self.restraint_frc_index = self.system.addForce(restraint_frc)
            

    
    # def set_restraint_positions(self,sytem, positions, atomlist):
    #     """
    #     Set reference positions for energy restraints.  This may be a different set of positions
    #     from the "current" positions that are stored in self.mol and self.xyz_omm.
    #     """
    #     if self.restraint_frc_index is not None:
            
    #         frc = system.getForce(self.restraint_frc_index)
    #         for d in atomlist:
    #         ## Generate OpenMM-compatible positions in nanometers
    #             for i, j in enumerate(np.arange(len(self.molecule.elem))):
    #                 xyz = positions[d]
    #                 print(d)
    #                 if i == d:
    #                     frc.setParticleParameters(int(i), int(j), xyz )
                        
    #             frc.updateParametersInContext(self.simulation.context)
    #     else:
    #         raise RuntimeError('Asked to set restraint positions, but no restraint force has been added to the system')


    def compute_energy(self, param=None, platform=None,justcomp=True,crit=1e-4):
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
        
        self.mm_energy = Quantity(value=np.zeros([self.n_frames], np.float64),unit=kilojoules_per_mole)
        
        if justcomp:
            for i in range(self.n_frames):
                self.context.setPositions(self.positions[i])
                state = self.context.getState(getEnergy=True)
                self.mm_energy[i] = state.getPotentialEnergy()
        else:
            

                
            steps = int(max(1, -1*np.log10(crit)))
            
            if self.gro:
                self.mm_opt_positions= np.ndarray((0, self.topology.getNumAtoms(), 3))
            else:
                self.mm_opt_positions= np.ndarray((0, self.topology.n_atoms, 3))
                    
            # print(np.ndarray((1, self.topology.n_atoms, 3)))
            for i in range(self.n_frames):
                integrator = mm.LangevinIntegrator(
                        300*kelvin,       # Temperature of heat bath
                        1.0/picoseconds,  # Friction coefficient
                        2.0*femtoseconds # Time step
                        )
                sys_f=deepcopy(self.system)
                sys_f= self.freeze_atoms(sys_f,self.torsion_index[i])
                simulation = app.Simulation(self.structure, sys_f, integrator, platform)
                simulation.context.setPositions(self.positions[i])
                for logc in np.linspace(0, np.log10(crit), steps):
                    simulation.minimizeEnergy( maxIterations=100000)
                for _ in range(1000):
                    e_minimized = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
                    simulation.minimizeEnergy(maxIterations=10)
                    e_new = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
                    if abs(e_new - e_minimized) < crit * 10:
                        break
                else:
                    raise RuntimeError("Energy minimization did not converge")
                
                state = simulation.context.getState(getEnergy=True,getPositions=True)
                self.mm_energy[i] = state.getPotentialEnergy()
                pos_new= [ [i.x,i.y,i.z]for i in state.getPositions() ]
                self.mm_opt_positions=np.append(self.mm_opt_positions, [pos_new], axis=0 ) 
                

        self.energy_unit = self.energy_unit



    def compute_minimized_structure(self,xyz,param=None,plot=False,platform=None):
        
        """ Computes minimized energy structure for a given parameter set

        Parameters
        ----------
        offset :
        platform: simtk.openmm.Platform to evaluate energy on (if None, will select automatically)
        """

        # Check if context exists.
        if not self.context:
            self.create_context(param, platform)
        else:
            # copy new torsion parameters
            self.copy_torsions(param, platform)

        mol= Molecule(xyz)
        pos= mol.Data['xyzs'][0]
        # Compute potential energies for all snapshots.
        # pos= deepcopy(self.positions[0])
        
        integrator = mm.LangevinIntegrator(
                        300*kelvin,       # Temperature of heat bath
                        1.0/picoseconds,  # Friction coefficient
                        2.0*femtoseconds # Time step
                        )

        
        simulation = app.Simulation(self.structure, self.system, integrator, platform)
        simulation.context.setPositions(pos)
        energy_init = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        simulation.minimizeEnergy()
        state = simulation.context.getState(getEnergy=True,getPositions=True)
        
        self.mm_energy_min = state.getPotentialEnergy()
        
        pos_new= [ [i.x,i.y,i.z]for i in state.getPositions() ]
        
        
        out = Output()
        def extract(b):
            with out:
                frm=molview2.frame
                f=open("mm_minimized_coord.pdb", "w")
                print(MyStructureTrajectory(mol).get_structure_string(), file=f)
                f.close()
                print("file written")
        
        
        if plt:
            grid = GridspecLayout(1, 2)  
            mol_init= deepcopy(mol)
            mol.Data['xyzs'][0]= np.array(pos_new)*10
               
            molview1=nglview.NGLWidget(MyStructureTrajectory(mol_init))
            text1 = ipywidgets.Button(description= "initial",
                                      layout=Layout(width='800px', grid_area='header'),
                                      style=ButtonStyle(button_color='lightblue',font_size="35",font_weight='bold') ) 
            text_1 = ipywidgets.Button(description= "Energy:%f"%energy_init.__getstate__()['_value'],
                                layout=Layout(width='800px', grid_area='header'),
                                style=ButtonStyle(button_color='lightblue',font_size="35",font_weight='bold') )   
                
            molview2=nglview.NGLWidget(MyStructureTrajectory(mol)) 
            text2 = ipywidgets.Button(description= "new",
                                      layout=Layout(width='800px', grid_area='header'),
                                      style=ButtonStyle(button_color='lightblue',font_size="35",font_weight='bold') ) 
            
            text_2 = ipywidgets.Button(description= "Energy:%f"%self.mm_energy_min.__getstate__()['_value'],
                    layout=Layout(width='800px', grid_area='header'),
                    style=ButtonStyle(button_color='lightblue',font_size="35",font_weight='bold') )   
           
            Button1 = ipywidgets.Button(description= "",
                    layout=Layout(width='800px', grid_area='header'),
                    style=ButtonStyle(button_color='orange',font_size="35",font_weight='bold'))
                
            Button2 = ipywidgets.Button(description= "Extract Frame as .xyz",
                    layout=Layout(width='800px', grid_area='header'),
                    style=ButtonStyle(button_color='orange',font_size="35",font_weight='bold'))
            Button2.on_click(extract)
            
            molview1._set_size('800px', '500px')
            molview2._set_size('800px', '500px')
            grid[0,0]=ipywidgets.VBox([text1,Button1, molview1,text_1])
            grid[0,1]=ipywidgets.VBox([text2,Button2, molview2,text_2])
            display.display(grid)
            
            
            



            
            
            
            

    def plot_qm_energies(self,save=False, plot=True, mol=True):
        
        increment=360/self.n_frames
        molecule= self.molecule
        
        molecule.comms=['' for i in range(self.n_frames)] # readjust after slice
        
        
        grid = GridspecLayout(1, 2)
        if len(self.d_name.keys()) == 1:  
            d_name= str(*self.d_name.keys())
            d_num=self.d_name[d_name]
            
        else:
            dna_val=[*self.d_name.values()]
            dna_keys=[*self.d_name.keys()]
            d_nums,dtor =np.transpose(dna_val)
            # print(dna_keys,dtor)
            d_num= d_nums[0]
            
        m_all, goAtoms = molecule.rotate_bond(0,*d_num[1:3],increment=increment)
        molecule.Data['xyzs']=np.copy(self.positions)*10
        Di_con=self.angles
        E_qm= np.copy(self.qm_energy)
         
        

        out = Output()
                
        # from itertools import cycle
        # color = cycle(['teal', 'goldenrod', 'aquamarine','tomato'])
        color = sns.color_palette("flare", n_colors=len(self.d_name.keys()))
        sns.set_palette(color) 
        fig, ax = plt.subplots() #constrained_layout=True ) 
        ax.set_xlabel('$ \psi $ / °') # X axis data label 
        ax.set_ylabel(r'$E_{pot}$ / '+self.energy_unit.get_name() )

        with out:
            fig.canvas.toolbar_position = 'bottom'
            n=0
            m=0
            for dt in dtor: 
                o=n
                n+=dt
                ax.plot(Di_con[range(o,n)],E_qm[range(o,n)], marker="h" , label='%s'%dna_keys[m] )
                m+=1
                
            line1, = ax.plot(Di_con[0],E_qm[0], marker="h",c='black' )
            plt.legend(loc= 'center', bbox_to_anchor=(1.4,0.5))
            plt.show()

        
        def on_value_change(b):
            with out:
                try:
                    frm=b['new']
                    
                    n=0
                    for dt in dtor: 
                        o=n
                        n+=dt
                        ax.plot(Di_con[range(o,n)],E_qm[range(o,n)], marker="h",c=next(color) )
                    ax.plot(Di_con[frm],E_qm[frm], marker="h" ,c='black')
                    

                except:
                    print()
        
        def extract(b):
            with out:
                frm=molview.frame
                f=open("new_coord.xyz", "w")
                print(MyStructureTrajectory(molecule).get_structure_string_xyz(frm), file=f)
                f.close()
                print("file written")
        
        
        Button = ipywidgets.Button(description= "Extract Frame as .xyz",
                    layout=Layout(width='800px', grid_area='header'),
                    style=ButtonStyle(button_color='lightblue',font_size="35",font_weight='bold'))
        if self.mm_energy is not None:
            text = ipywidgets.Button(description= " ",
                        layout=Layout(width='800px', grid_area='header'),
                        style=ButtonStyle(button_color='lightblue',font_size="35",font_weight='bold'))
            
        molview = nglview.NGLWidget(MyStructureTrajectory(molecule)) #._set_size(200,200)   
        molview._set_size('800px', '500px')
        
        if mol:
            grid[0,0]= ipywidgets.VBox([text,molview])
        if plot:
            grid[0,1]= ipywidgets.VBox([Button,out])
        
        molview.observe(on_value_change, ['frame'])
        Button.on_click(extract)

        # if save:
        #     plt.savefig( "%s.jpeg"%(d_name),dpi=200)
        
        return   display.display(grid)



    def plot_torsions(self,save=False, plot=True, mol=True):
        
        try:
            d_name= str(*self.d_name.keys())
            d_num=self.d_name[d_name]
        
        except:
            warnings.warn("Warning: Only Torsion sliced Object can be plotted")
            
        increment=360/self.n_frames
        
        # for self in self.slice_by_trosions():
            
        molecule= self.molecule
        
        molecule.comms=['' for i in range(self.n_frames)] # readjust after slice
        
        
        grid = GridspecLayout(1, 2)  
        d_name= str(*self.d_name.keys())
        d_num=self.d_name[d_name][0]

        m_all, goAtoms = molecule.rotate_bond(0,*d_num[1:3],increment=10)
        molecule.Data['xyzs']=np.copy(self.positions)*10
        Di_con=self.angles
        E_qm= np.copy(self.qm_energy) #-self.qm_energy.min())
        E_mm=np.copy(self.mm_energy) #-self.mm_energy.min()) 
         

    
        out = Output()
        fig, ax = plt.subplots(constrained_layout=True ) 
        ax.set_xlabel('$ \psi $ / °') # X axis data label 
        ax.set_ylabel(r'$E_{pot}$ / '+self.energy_unit.get_name() )
        
        
        with out:
            fig.canvas.toolbar_position = 'bottom'
            ax.plot(Di_con,E_qm, marker="h",c='teal', label='qm energies' )
            line1, = ax.plot(Di_con[0],E_qm[0], marker="h",c='orange' )
            if self.mm_energy is not None:
                ax.plot(Di_con, E_mm , marker="h",c='red', label='mm energies' ,)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.2))
            plt.show()

        
        def on_value_change(b):
            with out:
                try:
                    frm=b['new']
                    ax.plot(Di_con,E_qm, marker="h",c='teal')
                    ax.plot(Di_con[frm],E_qm[frm], marker="h",c='orange')

                except:
                    print()
        
        def extract(b):
            with out:
                frm=molview.frame
                # f=open("%s/new_coord.xyz"%(un_opt), "w")
                # print(MyStructureTrajectory(molecule).get_structure_string_xyz(frm), file=f)
                # f.close()
                # print("file written")
        
        
        Button = ipywidgets.Button(description= "Extract Frame as .xyz",
                    layout=Layout(width='800px', grid_area='header'),
                    style=ButtonStyle(button_color='lightblue',font_size="35",font_weight='bold'))
        
        text = ipywidgets.Button(description= "%s"%(d_name),
                        layout=Layout(width='800px', grid_area='header'),
                        style=ButtonStyle(button_color='lightblue',font_size="35",font_weight='bold'))

        molview = nglview.NGLWidget(MyStructureTrajectory(molecule)) #._set_size(200,200)   
        molview._set_size('800px', '500px')
        
        if mol:
            grid[0,0]= ipywidgets.VBox([text,molview])
        if plot:
            grid[0,1]= ipywidgets.VBox([Button,out])
        
        molview.observe(on_value_change, ['frame'])
        Button.on_click(extract)

        if save:
            plt.savefig( "%s.jpeg"%(d_name),dpi=200)
        
        return   display.display(grid)

        

    def build_phis(self, calc=None):
        """
        This function builds a dictionary of phis for specified dihedrals in the molecules for all frames in the qm db.

        Parameters
        ----------
        to_optimize : list of dihedral types to calculate phis for
            Default is None. When None, it will calculate phis for all dihedral types in molecule

        """
        
        atomtypes=[i.name for i in self.structure.atoms]  
        d_name_values=[*self.d_name.values()]
        d_num=[d_name_values[i][0] for i in range(len(d_name_values))]
        # print(d_num)
        
        # d_name_at="%s %s %s %s"%(atomtypes[d_num1[0]], atomtypes[d_num1[1]],atomtypes[d_num1[2]],atomtypes[d_num1[3]] )
        d_name_at=["%s %s %s %s"%(atomtypes[i[0]], atomtypes[i[1]],atomtypes[i[2]],atomtypes[i[3]]) for i in d_num]
        # print(d_name_at)
        
        molecule= self.molecule
        if self.mm_opt_positions is not None:
            # print("ok")
            molecule.Data['xyzs']=self.mm_opt_positions*10 
        else:
            molecule.Data['xyzs']=self.positions*10 
            
        molecule.comms=['' for i in range(self.n_frames)]
        
        
        # self.phis = {dn:  {dn_at: [[ ] for i in range(self.d_name[dn][1])] for dn_at in d_name_at}  for dn in self.d_name.keys()}
        # n=0    
        # for phi1 in self.phis.keys():
        #     o=n
        #     d_name1= str(phi1)
        #     d_num1, d_frames1 =self.d_name[d_name1]
 
        #     # print(di_name)
        #     n+=d_frames1
        #     for phi2, dn_at in zip(self.phis.keys(),d_name_at):
        #         d_name2= str(phi2)
        #         d_num2, d_frames2 =self.d_name[d_name2]
        #         # print(d_num,d_frames)
        #         calc_dihedrals= np.round(molecule.measure_dihedrals(*d_num2),2)
        #         self.phis[phi1][dn_at]=calc_dihedrals[o:n]
        
        
        self.phis = {dn:  {dn_at: [[ ] for i in range(self.d_name[dn][1])] for dn_at in self.d_name.keys()}  for dn in self.d_name.keys()}
        n=0    
        for phi1 in self.phis.keys():
            o=n
            d_name1= str(phi1)
            d_num1, d_frames1 =self.d_name[d_name1]
 
            # print(di_name)
            n+=d_frames1
            for phi2, dn_at in zip(self.phis.keys(),d_name_at):
                d_name2= str(phi2)
                d_num2, d_frames2 =self.d_name[d_name2]
                # print(d_num,d_frames)
                calc_dihedrals= np.round(molecule.measure_dihedrals(*d_num2),2)
                self.phis[phi1][phi2]=calc_dihedrals[o:n]
                
                

            
            
           
            
        
    def to_dataframe(self, psi4=True):

        """ convert TorsionScanSet to pandas dataframe

        Parameters
        ----------
        psi4 : bool
            Flag if QM log file is from psi4. Default True.
        """


        if len(self.mm_energy) == self.n_frames:
            mm_energy = self.mm_energy
            qm_energy = self.qm_energy
        else:
            mm_energy = [float('nan') for _ in range(self.n_frames)]
            
        data=[( np.round(qm_energy[i],4),  np.round(mm_energy[i],4) ) for i in range(self.n_frames)]
        columns = [ 'QM energy', 'MM energy ']
        energy_df = pd.DataFrame(data, columns=columns)
        
        if self.phis != None:
            phi_df=pd.DataFrame.from_dict(self.phis)
        #     torsion_set = pd.concat([phi_df, energy_df], axis=1, join="inner")
        # else:
        #     torsion_set = energy_df
        
        return phi_df, energy_df #torsion_set
  
        


    def remove_nonoptimized(self):
        """
        Remove configurations where optimizer failed
        Returns: copy of scan set with only optimized structures

        Returns
        -------
        new QMDataBase

        """
        key = []
        for i, optimized in enumerate(self.optimized):
            if optimized:
                key.append(i)
        new_torsionscanset = self.slice(key)
        return new_torsionscanset

    def remove_bad(self, bad_key):
        """
        Remove specific configurations
        Returns: copy of scan set with out key
        Returns
        -------
        new QMDataBase

        """
        key = []
        for i in range(self.n_frames):
            if i not in bad_key: 
                key.append(i)
        new_torsionscanset = self.slice(key)
        
        return new_torsionscanset

    

    @property
    def _have_mm_energy(self):
        return len(self.mm_energy) != 0



    def slice_by_trosions(self):
        sliced=[]
        n=0
        for dn in self.d_name.keys():
            o=n
            n+=self.d_name[dn][1]
            sliced.append(self.slice(range(o,n), slice_by_torsions=dn ) )
        return sliced

    
    def __getitem__(self, key):
        "Get a slice of this trajectory"
        return self.slice(key)
    

    def combine(self, dihedral_list):
        """merge dihedrals together
        """
        
        molecule= self.molecule
        structure = self.structure
        topology = self.topology
        gro=self.gro
        energy_unit= self.energy_unit
        
        xyz = deepcopy( self.xyz)
        qm_energy = deepcopy( self.qm_energy)
        mm_energy = deepcopy( self.mm_energy)
        angles = deepcopy( self.angles)
        d_name = deepcopy( self.d_name)
        torsions = deepcopy(self.torsion_index)
        mm_opt_position = self.mm_opt_positions
        
        for dih in dihedral_list:
            xyz=np.append(xyz, dih.xyz,axis=0)
            mm_opt_position =  np.append(mm_opt_position, dih.mm_opt_positions, axis=0)
            qm_energy = np.append(qm_energy, dih.qm_energy)
            mm_energy = np.append(mm_energy, dih.mm_energy)
            angles = np.append(angles, dih.angles)
            torsions =  np.append(torsions, dih.torsion_index, axis=0)
            
            
            d_name = d_name | dih.d_name
            
        xyz=Quantity(value=xyz, unit=nanometers)
        qm_energy=Quantity(value=qm_energy, unit=self.energy_unit)  
        mm_energy=Quantity(value=mm_energy, unit=self.energy_unit)  
             
 
        newtraj = self.__class__( molecule=molecule, positions=xyz, topology=topology, structure=structure, 
                                    qm_energies=qm_energy,torsion_index=torsions, extracted_torsions=d_name, mm_energies=mm_energy, 
                                    mm_opt_positions=mm_opt_position, angles=angles, gro=gro, energy_unit=energy_unit)
        return newtraj       



   

    def slice(self, key, copy=True, slice_by_torsions=None):
        """Slice trajectory, by extracting one or more frames into a separate object

        This method can also be called using index bracket notation, i.e
        `traj[1] == traj.slice(1)`

        Parameters
        ----------
        key : {int, np.ndarray, slice}
            The slice to take. Can be either an int, a list of ints, or a slice
            object.
        copy : bool, default=True
            Copy the arrays after slicing. If you set this to false, then if
            you modify a slice, you'll modify the original array since they
            point to the same data.
        """
        xyz = self.xyz[key]
        molecule= self.molecule
        topology = self.topology
        time = self.time[key]
        gro=self.gro
        torsions = self.torsion_index[key]
        qm_energy = self.qm_energy[key]
        
        if self._have_mm_energy:
            mm_energy = self.mm_energy[key]
            
        if self.mm_opt_positions is not None:
            mm_opt_position = self.mm_opt_positions[key]
        else:
            mm_opt_position = self.mm_opt_positions
            
        if slice_by_torsions != None:
            d_name = dict( [  (slice_by_torsions, (self.d_name[slice_by_torsions][0],len(torsions)) ) ])
        else:
            d_name = self.update_d_name(torsions)
        
        if self.phis != None and slice_by_torsions != None:
            phis= self.phis[slice_by_torsions]
        else:
            phis=self.phis
        
        if self.optimized is not None:
            optimized = self.optimized[key]

        if self.angles is not None:
            angles = self.angles[key]
        
        if copy:
            xyz = xyz.copy()
            time = time.copy()
            molecule= deepcopy(self.molecule)
            topology = deepcopy(self.topology)
            structure = deepcopy(self.structure)
            qm_energy = deepcopy(qm_energy)
            torsions = torsions.copy()
            
            if self._have_mm_energy:
                mm_energy = deepcopy(mm_energy) 
            else:
                mm_energy = self.mm_energy
                
            if self.mm_opt_positions is not None:
                mm_opt_position = deepcopy(mm_opt_position)
            else:
                mm_opt_position = self.mm_opt_positions
                
            if self.angles is not None:
                angles = angles.copy()
            else:
                angles = self.angles
                
            if self.optimized is not None:
                optimized = optimized.copy()
            else:
                optimized = self.optimized

        newtraj = self.__class__( molecule=molecule, positions=xyz, topology=topology, structure=structure, 
                                    qm_energies=qm_energy, phis=phis, extracted_torsions=d_name, mm_energies=mm_energy, 
                                    optimized=optimized, angles=angles, time=time,gro=gro, mm_opt_positions=mm_opt_position,
                                    torsion_index=torsions, energy_unit=self.energy_unit)

        # if self._rmsd_traces is not None:
        #     newtraj._rmsd_traces = np.array(self._rmsd_traces[key],
        #                                     ndmin=1, copy=True)
        return newtraj       


