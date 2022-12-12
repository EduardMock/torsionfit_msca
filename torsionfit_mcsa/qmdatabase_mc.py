__author__ = 'Chaya D. Stern'

import pandas as pd
import numpy as np

from simtk.unit import Quantity, nanometers, kilojoules_per_mole, kilocalories_per_mole, picoseconds, angstroms

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



def parse_psi4_out(oufiles_dir,xyz, structure, pattern="*.out"):
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
    if structure.endswith('psf'):
        topology = md.load_psf(structure)
        structure=parmed.load_file(structure)
        # structure = CharmmPsfFile(structure)
        
    else:
        topology = md.load(structure).topology
        structure = parmed.load_file(structure)


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
    # print(out_files)
    
    
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


    positions = np.ndarray((0, topology.n_atoms, 3))
    qm_energies = np.ndarray(0)
    torsions = np.ndarray((0, 4), dtype=int)
    angles = np.ndarray(0, dtype=float)
    optimized = np.ndarray(0, dtype=bool)



    # Parse files
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
            except AttributeError:
                warnings.warn("Warning: Check if the file terminated before completing SCF")
                qm_energy = np.array([np.nan])
        qm_energies = np.append(qm_energies, qm_energy, axis=0)
        qm_energies=Quantity(value=qm_energies, unit=kilojoules_per_mole)


    # Subtract lowest energy to find relative energies
    # qm_energies = qm_energies - min(qm_energies)
    
    angles = np.asarray(list(itertools.chain.from_iterable(dih_angles)))
    
    d_name=torsions.copy()
    _, idx, counts =np.unique(d_name,return_index=True,return_counts=True, axis=0)
    d_name_unique=d_name[np.sort(idx)]
    
    extracted_torsions=dict(zip(torsion_angles, zip(d_name_unique.tolist(),counts)  ))
    
    print(qm_energies)
    return QMDataBase(molecule=molecule, positions=positions, topology=topology, structure=structure, torsions=torsions, angles=angles,
                      qm_energies=qm_energies, optimized=optimized,extracted_torsions=extracted_torsions) 
        
    





class QMDataBase(DataBase):
    """container object for torsion scan

    A TorsionScanSet should be constructed by loading Gaussian 09 torsion scan log files or a psi4 output file from disk
    with an mdtraj.Topology object

    Attributes
    ----------
    structure: ParmEd.Structure
    qm_energy: simtk.unit.Quantity((n_frames), unit=kilojoule/mole)
    mm_energy: simtk.unit.Quantity((n_frames), unit=kilojoule/mole)
    delta_energy: simtk.unit.Quantity((n_frames), unit=kilojoule/mole)
    torsion_index: {np.ndarray, shape(n_frames, 4)}
    step: {np.ndarray, shape(n_frame, 3)}
    direction: {np.ndarray, shape(n_frame)}. 0 = negative, 1 = positive
    """

    def __init__(self,molecule, positions, topology, structure, torsions, qm_energies,mm_energies=Quantity(), angles=None, optimized=None, directions= None, steps=None, extracted_torsions=None,time=None):
        """Create new TorsionScanSet object"""
        assert isinstance(topology, object)
        super(QMDataBase, self).__init__(positions, topology, structure, time)
        self._energy_unit = kilojoules_per_mole
        
        self.molecule=molecule
        self.qm_energy = qm_energies
        self.mm_energy = mm_energies
        self.initial_mm = Quantity()
        self.delta_energy = Quantity()
        self.d_name= extracted_torsions
        self.torsion_index = torsions
        self.directions = directions
        self.steps = steps
        self.angles = angles
        self.optimized = optimized
        self.phis = {}

    @property
    def energy_unit(self):
        return self._energy_unit   

    @energy_unit.setter
    def energy_unit(self, unit):
        """Change energy to choosen unit"""
        self._energy_unit = unit 
        self.qm_energy = self.qm_energy.in_units_of(unit)
        self.mm_energy = self.mm_energy.in_units_of(unit)

    
    def compute_energy(self, param, offset=None, platform=None):
        """ Computes energy for a given structure with a given parameter set

        Parameters
        ----------
        param: parmed.charmm.CharmmParameterSet
        platform: simtk.openmm.Platform to evaluate energy on (if None, will select automatically)
        """

        # Save initial mm energy
        save = False
        if not self._have_mm_energy:
            save = True

        # calculate energy
        super(QMDataBase, self).compute_energy(param, platform)

        # Subtract off minimum of mm_energy and add offset
        

        # min_energy = self.mm_energy.min()
        # self.mm_energy -= min_energy
        if save:
            self.initial_mm = deepcopy(self.mm_energy)
        if offset:
            offset = Quantity(value=offset.value, unit=self.mm_energy.unit)
            self.mm_energy += offset
            
        # self.delta_energy = (self.qm_energy - self.mm_energy)
        # self.delta_energy = self.delta_energy - self.delta_energy.min()



    def to_dataframe(self, psi4=True):

        """ convert TorsionScanSet to pandas dataframe

        Parameters
        ----------
        psi4 : bool
            Flag if QM log file is from psi4. Default True.
        """

        if len(self.mm_energy) == self.n_frames and len(self.delta_energy) == self.n_frames:
            mm_energy = self.mm_energy
            delta_energy = self.delta_energy
        else:
            mm_energy = [float('nan') for _ in range(self.n_frames)]
            delta_energy = [float('nan') for _ in range(self.n_frames)]
        if psi4:
            data = [(self.torsion_index[i], self.angles[i], self.qm_energy[i], mm_energy[i], delta_energy[i],
                    self.optimized[i]) for i in range(self.n_frames)]

            columns = ['Torsion', 'Torsion angle', 'QM energy (KJ/mol)', 'MM energy (KJ/mol)', 'Delta energy (KJ/mol)',
                       'Optimized']
        # else:
        #     data = [(self.torsion_index[i], self.directions[i], self.steps[i], self.qm_energy[i], mm_energy[i],
        #              delta_energy[i]) for i in range(self.n_frames)]
        #     columns = ['Torsion', 'direction','steps', 'QM energy (KJ/mol)', 'MM energy (KJ/mol)',
        #                'Delta energy (KJ/mol)']

        torsion_set = pd.DataFrame(data, columns=columns)
        return torsion_set




    def plot_torsions(self):
        
        try:
            d_name= str(*self.d_name.keys())
            d_num=self.d_name[d_name]
        
        except:
            warnings.warn("Warning: Only sliced Object can be plotted")
            
        increment=360/self.n_frames
        
        # for self in self.slice_by_trosions():
            
        molecule= self.molecule
        
        molecule.comms=['' for i in range(self.n_frames)] # essential after sliceing 
        
        grid = GridspecLayout(1, 2)  
        d_name= str(*self.d_name.keys())
        d_num=self.d_name[d_name]

        # print(d_num)
        m_all, goAtoms = molecule.rotate_bond(0,*d_num[1:3],increment=10)
        molecule.Data['xyzs']=self.positions*10
        Di_con=self.angles
        # print(Di_con)
        E_mp2_con_rel= np.copy(self.qm_energy-self.qm_energy.min())
        
        # E_mp2_con_rel= E_mp2_con -np.min(E_mp2_con) 
         

    
        out = Output()
        fig, ax = plt.subplots(constrained_layout=True ) 
        ax.set_xlabel('$ \psi $ / °') # X axis data label 
        ax.set_ylabel(r'$E_{pot}$ /kcal$\cdot$mol$^{-1}$ ')
        
        
        with out:
            fig.canvas.toolbar_position = 'bottom'
            ax.plot(Di_con,E_mp2_con_rel, marker="h",c='teal', label='qm energies' )
            line1, = ax.plot(Di_con[0],E_mp2_con_rel[0], marker="h",c='orange' )
            if self.mm_energy is not None:
                ax.plot(Di_con, (self.mm_energy-self.mm_energy.min()) , marker="h",c='red', label='qm energies' )
            plt.show()

        def on_value_change(b):
            with out:
                try:
                    frm=b['new']
                    ax.plot(Di_con,E_mp2_con_rel, marker="h",c='teal')
                    ax.plot(Di_con[frm],E_mp2_con_rel[frm], marker="h",c='orange')

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
        
        grid[0,0]= ipywidgets.VBox([text,molview])
        grid[0,1]= ipywidgets.VBox([Button,out])
        
        molview.observe(on_value_change, ['frame'])
        Button.on_click(extract)

        return   display.display(grid)

        




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
            if i is not bad_key: 
                key.append(i)
        # print(key)
        new_torsionscanset = self.slice(key)
        return new_torsionscanset

    

    @property
    def _have_mm_energy(self):
        return len(self.mm_energy) != 0

    # def build_phis(self, calc=None):
    #     """
    #     This function builds a dictionary of phis for specified dihedrals in the molecules for all frames in the qm db.

    #     Parameters
    #     ----------
    #     to_optimize : list of dihedral types to calculate phis for
    #         Default is None. When None, it will calculate phis for all dihedral types in molecule

    #     """


    #     # print(calc)
    #     self.phis = {dn: [[] for i in range(sellf.d_name[dn][1])] for dn in self.d_name.keys()}
        
        
        
        

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
        time = self.time[key]
        torsions = self.torsion_index[key]
        

        
        if self.directions != None:
            directions = self.directions[key]
            
        if self.steps != None:
            steps = self.steps[key]
        
        if slice_by_torsions != None:
            d_name = dict(zip([slice_by_torsions],self.d_name[slice_by_torsions]))
        else:
            d_name = self.d_name
        
        if self.optimized is not None:
            optimized = self.optimized[key]

        if self.angles is not None:
            angles = self.angles[key]
            
        qm_energy = self.qm_energy[key]
        # print(qm_energy)
        mm_energy = self.mm_energy[key]
        unitcell_lengths, unitcell_angles = None, None
        
        if self.unitcell_angles is not None:
            unitcell_angles = self.unitcell_angles[key]
        
        if self.unitcell_lengths is not None:
            unitcell_lengths = self.unitcell_lengths[key]

        if copy:
            xyz = xyz.copy()
            time = time.copy()
            topology = deepcopy(self._topology)
            structure = deepcopy(self.structure)
            qm_energy = deepcopy(qm_energy)
            mm_energy = deepcopy(mm_energy)
            torsions = torsions.copy()
 
            
            if self.directions is not None:
                directions = directions.copy()
            else:
                directions = self.directions
            if self.optimized is not None:
                    optimized = optimized.copy()
            else:
                optimized = self.optimized
            if self.steps is not None:
                steps = steps.copy()
            else:
                steps = self.steps
            if self.angles is not None:
                angles = angles.copy()
            else:
                angles = self.angles
            if self.unitcell_angles is not None:
                unitcell_angles = unitcell_angles.copy()
            if self.unitcell_lengths is not None:
                unitcell_lengths = unitcell_lengths.copy()


        newtraj = self.__class__( molecule= molecule,
            positions=xyz, topology=topology, structure=structure, torsions=torsions, directions=directions, steps=steps, extracted_torsions=d_name, 
            qm_energies=qm_energy,mm_energies=mm_energy, optimized=optimized, angles=angles, time=time)

        if self._rmsd_traces is not None:
            newtraj._rmsd_traces = np.array(self._rmsd_traces[key],
                                            ndmin=1, copy=True)
        return newtraj       





        


# ---------------------------------------------------------------------------------------


    # def extract_geom_opt(self):
    #     """
    #     Extracts optimized geometry for Gaussian torsion scan.

    #     Returns
    #     -------
    #     New QMDataBase

    #     """
    #     key = []
    #     for i, step in enumerate(self.steps):
    #         try:
    #             if step[1] != self.steps[i+1][1]:
    #                 key.append(i)
    #         except IndexError:
    #             key.append(i)
    #     new_torsionScanSet = self.slice(key)
    #     return new_torsionScanSet
    



# def plot_torsions(torsion_psi4,xyz,structure,di_name,di_num,increment, shift_first=False, set_oo=True):
    
#     molecule= Molecule(xyz)
    
#     if structure.endswith('psf'):
#         topology = md.load_psf(structure)
#         structure=parmed.load_file(structure)
#         # structure = CharmmPsfFile(structure)
        
#     else:
#         topology = md.load(structure).topology
#         structure = parmed.load_file(structure)
    
    
    
#     grid = GridspecLayout(1, 2)   
#     m_all, goAtoms = molecule.rotate_bond(0,*di_num[1:3],increment=increment)
#     m_rot=m_all[::]
#     m_d=m_rot.Data['xyzs']
#     rot_by=m_rot.measure_dihedrals(*di_num)

#     m=0
#     E_mp2_con=[]  #kcal/mol
#     Di_con=[]
#     for i in range(len(m_rot[:])):
#         # print(i)
#         try:
#             rot_int= int(np.round(rot_by[i],1))  #+ increment*i    #
#             result="%s/%s/opt_tmp/gid_%i/result.dat"%(torsion_psi4,di_name,rot_int)
#             geoms="%s/%s/opt_tmp/gid_%i/geoms.xyz"%(torsion_psi4,di_name,rot_int)
#             geom_all=Molecule(geoms)
#             # print(m_rot[i].Data['xyzs'] )
#             # print( geom_all[-1].Data['xyzs'] )
#             m_rot.Data['xyzs'][i] = geom_all.Data['xyzs'][-1]

#             E_mp2= np.loadtxt(result, usecols=1, dtype=str)
#             Di_con.append(rot_int)
#             E_mp2_con.append(float(E_mp2))
            
#             #print(rot_int, E_mp2)
#             m+=1
#         except:
#             print()


    
    
#     if shift_first:
#         Di_con[0]=-int(np.round(rot_by[0],1))
#         Di_con.append(180) #np.insert(Di_con,-1,np.round(rot_by[0],1))
#         E_mp2_con=np.insert(E_mp2_con,0,E_mp2_con[-1])
#         #E_mp2_con=np.delete(E_mp2_con, -1)
        
#     if set_oo:
#         index= Di_con.index(min(Di_con))
#         print(index)
#         tmp1= Di_con[index:]
#         #print(tmp1)
#         tmp2=Di_con[:index]
#         #print(tmp2)
#         Di_con=tmp1+tmp2
#     # print(Di_con,E_mp2_con)

#     angles = np.asarray(Di_con)
#     torsions= di_num
#     qm_energies = Quantity(value=np.asarray(E_mp2_con), unit=kilojoules_per_mole)
#     positions = np.asarray(m_rot.Data['xyzs'] )*0.1  # in nanometers  
    
    
    
#     # self.n_frames = len(self.positions)
    
    
#     E_mp2_con_rel=E_mp2_con-np.min(E_mp2_con)  

    
#     out = Output()
#     fig, ax = plt.subplots(constrained_layout=True ) 
#     ax.set_xlabel('$ \psi $ / °') # X axis data label 
#     ax.set_ylabel(r'$E_{pot}$ /kcal$\cdot$mol$^{-1}$ ')
    
    
#     with out:
#         fig.canvas.toolbar_position = 'bottom'
#         ax.plot(Di_con,E_mp2_con_rel, marker="h",c='teal', label='qm energies' )
#         line1, = ax.plot(Di_con[0],E_mp2_con_rel[0], marker="h",c='orange' )
#         plt.show()

#     def on_value_change(b):
#         with out:
#             try:
#                 frm=b['new']
#                 ax.plot(Di_con,E_mp2_con_rel, marker="h",c='teal')
#                 ax.plot(Di_con[frm],E_mp2_con_rel[frm], marker="h",c='orange')

#             except:
#                 print()
    
#     def extract(b):
#         with out:
#             frm=molview.frame
#             f=open("%s/new_coord.xyz"%(un_opt), "w")
#             print(MyStructureTrajectory(m_rot).get_structure_string_xyz(frm), file=f)
#             f.close()
#             print("file written")
    
    
#     Button = ipywidgets.Button(description= "Extract Frame as .xyz",
#                 layout=Layout(width='800px', grid_area='header'),
#                 style=ButtonStyle(button_color='lightblue',font_size="35",font_weight='bold'))
    
#     text = ipywidgets.Button(description= "%s"%di_name,
#                     layout=Layout(width='800px', grid_area='header'),
#                     style=ButtonStyle(button_color='lightblue',font_size="35",font_weight='bold'))

#     molview = nglview.NGLWidget(MyStructureTrajectory(m_rot)) #._set_size(200,200)   
#     molview._set_size('800px', '500px')
    
#     grid[0,0]= ipywidgets.VBox([text,molview])
#     grid[0,1]= ipywidgets.VBox([Button,out])
    
#     molview.observe(on_value_change, ['frame'])
#     Button.on_click(extract)
#     # molview.observe(on_click, ['frame'])

#     return    QMDataBase(molecule=molecule, positions=positions, topology=topology, structure=structure, torsions=torsions,  qm_energies=qm_energies, angles=angles
#                       ,extracted_torsions=torsion_angles)
    
#            #display.display(grid)



