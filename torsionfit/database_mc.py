
import numpy as np
import pandas as pd

import simtk.openmm as mm
from simtk.unit import *
# from simtk.openmm.app import *   # not compatible

# from simtk.unit import Quantity, nanometers, angstroms, kilojoules_per_mole, kilocalorie_per_mole, picoseconds
from  mdtraj import Trajectory

from torsionfit import parameters as par

import matplotlib.pyplot as plt
import ipywidgets 
import IPython.display as display
from ipywidgets import Output, Tab, GridspecLayout, Layout, ButtonStyle
from IPython.display import clear_output

from .utils import *


from copy import deepcopy
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

    def __init__(self, pos, topology, structure, time=None):
        """Create new TorsionScanSet object"""
        assert isinstance(topology, object)
        super(DataBase, self).__init__(pos, topology, time)
        
        self.structure = structure
        self.positions = pos
        self.mm_energies = Quantity()
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
        
        

    def create_context(self, param, platform=None):
        """

        Parameters
        ----------
        param :
        platform :
        """
        self.structure.load_parameters(param)
        self.system = self.structure.createSystem(param,nonbondedMethod= mm.app.NoCutoff, constraints=None)
        self.integrator=mm.VerletIntegrator(0.004*picoseconds)
        if platform != None:
            self.context = mm.Context(self.system, self.integrator, platform)
        else:
            self.context = mm.Context(self.system, self.integrator)
 



    def copy_torsions(self, param=None, platform=None):
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



    def compute_energy(self, param, platform=None):
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
        for i in range(self.n_frames):
            self.context.setPositions(self.positions[i])
            state = self.context.getState(getEnergy=True)
            print(state.getPotentialEnergy())#.in_units_of(kilocalories_per_mole))
            self.mm_energy[i] = state.getPotentialEnergy()
        # print(self.mm_energy)






                
    
    def show_torsions(self, xyz, di_num, di_name= None, increment=30, ):
        
        molecule=Molecule(xyz)
        
        row= int(len(di_num)/4)
        if len(di_num)%4 > 0:
            row +=1
            
        grid = GridspecLayout(row, 4)

        
        if di_name is None:
            atomtypes=[i.type for i in self.structure.atoms]    
            di_name=["%s_%s_%s_%s"%(atomtypes[i[0]], atomtypes[i[1]],atomtypes[i[2]],atomtypes[i[3]]) for i in di_num]
            # conv.append([ [atomtypes[i[0]], atomtypes[i[1]],atomtypes[i[2]],atomtypes[i[3]]] for i in di_num])

        di_con=[]
        c=0
        r=0
        o=0
        for d in di_num:
            m_all, goAtoms = molecule.rotate_bond(0,*d[1:3],increment=increment)
            m_rot=m_all[::]
            molview = nglview.NGLWidget(MyStructureTrajectory(m_rot)) #._set_size(200,200)
            molview._set_size('500px', '500px')
            text = ipywidgets.Button(description= "%s"%di_name[o],
                        layout=Layout(width='500px', grid_area='header'),
                        style=ButtonStyle(button_color='lightblue',font_size="35",font_weight='bold'))
            grid[r,c]= ipywidgets.VBox([text,molview])
            c+=1
            o+=1
            if c==4:
                r+=1
                c=0
        
        
        return display.display(grid)




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



# ---------------------------------------------------------------------------------







    # def mm_from_param_sample(self, param, db, start=0, end=-1, decouple_n=False, phase=False, n_5=True, model_type='openmm'):

    #     """
    #     This function computes mm_energy for scan using sampled torsions
    #     Args:
    #         param: parmed.charmm.parameterset
    #         db: sqlit_plus database
    #         start: int, start of mcmc chain. Defualt 0
    #         end: int, end of mcmc chain. Default -1
    #         decouple_n: flag if multiplicities were sampled
    #         phase: flag if phases were sampled
    #         n_5: flag if multiplicity of 5 was sampled

    #     Returns:

    #     """
    #     N = len(db.sigma[start:end])
    #     mm_energy = np.zeros((N, self.n_frames))
    #     param_list = db.get_sampled_torsions()
    #     for i in range(N):
    #         par.update_param_from_sample(param_list, param, db=db,  i=i, rj=decouple_n, phase=phase, n_5=n_5,
    #                                      model_type=model_type)
    #         self.compute_energy(param)
    #         mm_energy[i] = self.mm_energy._value

    #     return mm_energy
    
    
    
    
    
    # def build_phis(self, calc=None):
    #     """
    #     This function builds a dictionary of phis for specified dihedrals in the molecules for all frames in the qm db.

    #     Parameters
    #     ----------
    #     to_optimize : list of dihedral types to calculate phis for
    #         Default is None. When None, it will calculate phis for all dihedral types in molecule

    #     """
    #     print(calc)
    #     # if calc is None:
    #     #     calc= self.di_name
                

    #     # print(calc)
    #     # self.phis = {dname: [[] for i in range(self.n_frames)] for dname in calc}
        

    
    
    # def to_dataframe(self, psi4=True):

    #     """ convert TorsionScanSet to pandas dataframe

    #     Parameters
    #     ----------
    #     psi4 : bool
    #         Flag if QM log file is from psi4. Default True.
    #     """

    #     if len(self.mm_energy) == len(self.positions) and len(self.delta_energy) == len(self.positions):
    #         mm_energy = self.mm_energy
    #         delta_energy = self.delta_energy
    #         phi=self.phis
    #         dipole=self.mm_dipole
    #     else:
    #         mm_energy = [float('nan') for _ in range(len(self.positions))]
    #         delta_energy = [float('nan') for _ in range(len(self.positions))]
            
    #     if psi4:
    #         data = [(self.phis[i], self.qm_energy[i], mm_energy[i], delta_energy[i], dipole[i][-1] ) for i in range(len(self.positions))]

    #         columns = [ 'phi (Degree)' ,'QM energy (Kcal/mol)', 'MM energy (Kcal/mol)', 'Delta energy (Kcal/mol)', 'Total MM Dipole ']
    #     # else:
    #     #     data = [(self.torsion_index[i], self.direction[i], self.steps[i], self.qm_energy[i], mm_energy[i],
    #     #              delta_energy[i]) for i in range(self.n_frames)]
    #     #     columns = ['Torsion', 'direction','steps', 'QM energy (KJ/mol)', 'MM energy (KJ/mol)',
    #     #                'Delta energy (KJ/mol)']

    #     torsion_set = pd.DataFrame(data, columns=columns)
    #     return torsion_set
    
    
    
    
    
    


    # def compute_energy_from_qm(self, param, platform=None):
    #     """ Computes energy for a given structure with a given parameter set

    #     Parameters
    #     ----------
    #     offset :
    #     param: parmed.charmm.CharmmParameterSet
    #     platform: simtk.openmm.Platform to evaluate energy on (if None, will select automatically)
    #     """

    #     if self.n_frames == 0:
    #         raise Exception("self.n_frames = 0! There are no frames to compute energy for.")

    #     # Check if context exists.
    #     if not self.context:
    #         self.create_context(param, platform)
    #     else:
    #         # copy new torsion parameters
    #         self.copy_torsions(param, platform)

    #     # Compute potential energies for all snapshots.
    #     self.mm_energy = Quantity(value=np.zeros([len(self.positions)], np.float64), unit=kilocalorie_per_mole)
    #     # self.mm_dipole = np.zeros([len(self.positions)]) #Quantity(value=np.zeros([len(self.positions)], np.float64), unit=debye)
    #     for i in range(len(self.positions)):
    #         # print(i)
    #         # print(self.positions[i])
    #         self.context.setPositions(self.positions[i])
    #         # self.mm_dipole[i] = self.dipole()
    #         state = self.context.getState(getEnergy=True)
    #         self.mm_energy[i] = state.getPotentialEnergy()



 # def plot_qm_torsions(self,torsion_psi4,xyz,di_name,di_num,increment, shift_first=False, set_oo=True):
        
    #     self.molecule= Molecule(xyz)
    #     grid = GridspecLayout(1, 2)   
    #     m_all, goAtoms = self.molecule.rotate_bond(0,*di_num[1:3],increment=increment)
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

    #     self.phis = np.asarray(Di_con)
    #     self.qm_energy = Quantity(value=np.asarray(E_mp2_con), unit=kilojoules_per_mole)
    #     self.positions = np.asarray(m_rot.Data['xyzs'] )*0.1  # in nanometers  
    #     print(self.positions)
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

    #     return  display.display(grid)