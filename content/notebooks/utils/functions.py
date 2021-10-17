import json
import numpy as np
import os
import subprocess
import shutil
import matplotlib.pyplot as plt
from types import SimpleNamespace


def readCoords(f):
    """Read XYZ file and return as MRChem JSON friendly string."""
    with open(f) as file:
        return '\n'.join([line.strip() for line in file.readlines()[2:]])
    
def makeEnergyInput(world_prec=None, xyzfile=None, fname=None):
    """
    Write MRChem JSON input file for energy calculation.
    
    Parameters:
    -------------
    
    world_prec: precision to use
    xyzfile: path to xyzfile
    fname: name of the generated input file (without extension, .inp assumed)"""
    i = {
        'world_prec': world_prec,
        'world_unit': 'angstrom',
        'Molecule': {
            'charge': 0,
            'multiplicity': 1,
            'translate': True,
            'coords': readCoords(xyzfile)
        },
        'WaveFunction': {
            'method': 'pbe', 
            'restricted': True
        },
        'SCF': {
            'guess_type': 'sad_dz',
            'guess_prec': 1e-4,
            'kain': 5,
            'write_orbitals': True,
            'localize': True,
            'max_iter': 20
        }
    }
    
    with open(fname+'.inp', 'w') as f:
        json.dump(i, f, indent=2)
        
def makeNMRInput(world_prec=None, xyzfile=None, fname=None):
    """
    Write MRChem JSON input file for NMR calculation.
    
    Parameters:
    -------------
    
    world_prec: precision to use
    xyzfile: path to xyzfile
    fname: name of the generated input file (without extension, .inp assumed)"""
    i = {
        'world_prec': world_prec,
        'world_unit': 'angstrom',
        'Molecule': {
            'charge': 0,
            'multiplicity': 1,
            'translate': True,
            'coords': readCoords(xyzfile)
        },
        'WaveFunction': {
            'method': 'pbe', 
            'restricted': True
        },
        'SCF': {
            'guess_type': 'sad_dz',
            'guess_prec': 1e-4,
            'kain': 5,
            'write_orbitals': True,
            'localize': True,
            'max_iter': 20
        },
        'Properties': {
            'nmr_shielding': True
        },
        'Response': {
            'kain': 5,
            'max_iter': 20
        }
    }
    
    with open(fname+'.inp', 'w') as f:
        json.dump(i, f, indent=2)

def submit(nprocs=None, inputfile=None):
    """Make calc dir, move inputfile, and start the calculation.
    
    Parameters:
    -----------
    nprocs      : Number of OpenMP threads to use
    inputfile   : Name of input file (without extension)"""

    ROOT = os.path.abspath('')
    
    # Check if all arguments given
    assert not any([nprocs is None, inputfile is None]), 'Missing arguments'
    
    dest = os.path.join(ROOT, f'{inputfile}_calc')
    
    # Make calc dir
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
        
    # Check if input file exists
    if not os.path.exists(inputfile+'.inp'):
        print(f'Error: Input file <{inputfile}.inp> not found')
        return
    
    shutil.move(inputfile+'.inp', os.path.join(dest, inputfile+'.inp'))
    os.chdir(dest)
    os.environ['OMP_NUM_THREADS'] = str(nprocs)
    subprocess.call(['mrchem', '--json',  inputfile+'.inp'])
    os.chdir(ROOT)
    
    print('Calculation done!')



class MRChemOutput:
    """Convenience class for accessing data from an MRChem calculation.
    
    You can access the JSON data via the data attribute.
    For example, to get the final SCF total energy:
    

    []: calc = MRChemOutput('jobname.json')
        E_tot = calc.data['output']['properties']['scf_energy']['E_tot']
    
    Additionally, the JSON is loaded as a nested SimpleNamespace
    (stored in the ns attribute),
    and you can quickly navigate the dict by 'dotting' through
    the keys. Tab completion should work for this in the Jupyter
    environment:
    
    []: E_tot = calc.ns.output.properties.scf_energy.E_tot
    
    Known bug
    Some keys are not valid Python variable names,
    and you cannot access the levels below these keys by dotting.
    Workaround: use __.getattribute__(key) to access such keys.
    
    A few methods are also provided that serve as shortcuts
    for navigating the data dictionary, and for plotting
    SCF convergence data.
    """
    
    def __init__(self, jsonfile):
        """Parameters:
           jsonfile <str>: Path to MRChem output JSON file
        """
        self.file = jsonfile
        with open(self.file) as f:
            self.raw = f.read()
        self.data = json.loads(self.raw)
        self.ns = json.loads(self.raw, object_hook=lambda item: SimpleNamespace(**item))
        
    def normalTermination(self):
        return self.data['output']['success']
                        
    def getFinalSCFEnergy(self) -> float:
        """Return optimized SCF energy in Hartrees."""
        return self.data['output']['properties']['scf_energy']['E_tot']
    
    def getNMRShieldingTensorsDiagonalized(self) -> dict:
        """Return diagonalized total NMR shielding tensor."""
        prop = self.data['output']['properties']['nmr_shielding']
        return{key: np.asarray(prop[key]['diagonalized_tensor']) for key in prop.keys()}
    
    def getNMRShieldingTensors(self) -> dict:
        """Return total NMR shielding tensor."""
        prop = self.data['output']['properties']['nmr_shielding']
        return{key: np.asarray(prop[key]['tensor']).reshape((3, 3)) for key in prop.keys()}
    
    def getNMRShieldingTensorsParamagnetic(self) -> dict:
        """Return paramagnetic contribution to NMR shielding tensor."""
        prop = self.data['output']['properties']['nmr_shielding']
        return{key: np.asarray(prop[key]['tensor_para']).reshape((3, 3)) for key in prop.keys()}
    
    def getNMRShieldingTensorsDiamagnetic(self) -> dict:
        """Return diamagnetic contribution to NMR shielding tensor."""
        prop = self.data['output']['properties']['nmr_shielding']
        return{key: np.asarray(prop[key]['tensor_dia']).reshape((3, 3)) for key in prop.keys()}
              
    def getNMRShieldingIsotropicAverage(self) -> dict:
        """Return isotropic average of total NMR shielding tensor."""
        prop = self.data['output']['properties']['nmr_shielding']
        return{key: float(prop[key]['isotropic_average']) for key in prop.keys()}
    
    def getSCFConvergence(self) -> list:
        """Return list of tuples containing total energy, energy update, and MO residuals
        for all SCF iterations."""
        cycles = self.data['output']['scf_calculation']['scf_solver']['cycles']
        return [(cycle['energy_total'], cycle['energy_update'], cycle['mo_residual']) for cycle in cycles]
    
    def getNMRShieldingAnisotropy(self) -> dict:
        """Return anisotropy of the diagonalized NMR shielding tensor."""
        prop = self.data['output']['properties']['nmr_shielding']
        return{key: float(prop[key]['anisotropy']) for key in prop.keys()}
    
    def getResponseConvergence(self) -> list:
        """Return convergence data for the x, y, and z dimension response SCFs."""
        components = self.data['output']['rsp_calculations']['ext_mag-0.000000']['components']
        result = []
        for comp in components:
            cycles = comp['rsp_solver']['cycles']
            row = [(cycle['symmetric_property'], cycle['mo_residual'], cycle['property_update']) for cycle in cycles]
            result.append(row)
        return result
    
    def getWalltime(self):
        w_scf = self.data['output']['scf_calculation']['scf_solver']['wall_time']
        w_rsp = 0
        for field, d in self.data['output']['rsp_calculations'].items():
            for dim in d['components']:
                w_rsp += dim['rsp_solver']['wall_time']
        return w_scf, w_rsp
    
    def plotSCFConvergence(self):
        """Plot the SCF convergence in terms of total energy, energy udpates, and MO residuals."""
        e, u, mo = zip(*self.getSCFConvergence())
        xs = np.arange(len(e))
        
        prec = self.data['input']['scf_calculation']['scf_solver']['final_prec']
        thrs_up = self.data['input']['scf_calculation']['scf_solver']['energy_thrs']
        thrs_mo = self.data['input']['scf_calculation']['scf_solver']['orbital_thrs']
        
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax2 = ax.twinx()
        ax.set_yscale('log')
        
        ax.set_xlabel('SCF iterations')
        ax.set_ylabel('a.u.')
        ax2.set_ylabel('Total energy (a.u.)')
        ax.set_title(f'SCF Convergence for {self.file}')
        
        ax2.plot(xs, e, marker='.', color='black', mec='black', label='Total energy')
        ax.plot(xs, np.abs(u), marker='.', color='salmon', mec='black', label='Energy update')
        ax.plot(xs, np.abs(mo), marker='.', color='skyblue', mec='black', label='MO residual')
        
        if int(thrs_up) != -1:
            ax.axhline(thrs_up, ls='--', color='salmon', lw=2)
        if int(thrs_mo) != -1:
            ax.axhline(thrs_mo, ls='--', color='skyblue', lw=2)
        
        ax_h, ax_l = ax.get_legend_handles_labels()
        ax2_h, ax2_l = ax2.get_legend_handles_labels()
        
        ax.legend(ax_h+ax2_h, ax_l+ax2_l, fontsize='small')
        ax.grid()
        plt.tight_layout(h_pad=1)
        plt.show()
        
    def plotResponseConvergence(self):
        """Plot the response SCF convergence in terms of the property, property update, and MO residual.""" 
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax2 = ax.twinx()
        ax.set_yscale('log')
        
        ax.set_xlabel('SCF iterations')
        ax.set_ylabel('a.u.')
        ax2.set_ylabel('Property (a.u.)')
        ax.set_title(f'Response convergence for {self.file}')
        
        for i, dim in enumerate(self.getResponseConvergence()):
            p, u, mo = zip(*dim)
            xs = np.arange(len(p))
            
            prec = self.data['input']['rsp_calculations']['ext_mag-0.000000']['components'][i]['rsp_solver']['final_prec']
            thrs_pr = self.data['input']['rsp_calculations']['ext_mag-0.000000']['components'][i]['rsp_solver']['property_thrs']
            thrs_mo = self.data['input']['rsp_calculations']['ext_mag-0.000000']['components'][i]['rsp_solver']['orbital_thrs']
        
            ax2.plot(xs, p, marker='.', color='black', mec='black', label='Property' if i == 0 else None)
            ax.plot(xs, np.abs(u), marker='.', color='salmon', mec='black', label='Property update' if i == 0 else None)
            ax.plot(xs, np.abs(mo), marker='.', color='skyblue', mec='black', label='MO residual' if i == 0 else None)
            
            if int(thrs_pr) != -1:
                ax.axhline(thrs_pr, ls='--', color='salmon', lw=2)
            if int(thrs_mo) != -1:
                ax.axhline(thrs_mo, ls='--', color='skyblue', lw=2)
        
        ax_h, ax_l = ax.get_legend_handles_labels()
        ax2_h, ax2_l = ax2.get_legend_handles_labels()
        
        ax.legend(ax_h+ax2_h, ax_l+ax2_l, fontsize='small')
        ax.grid()
        plt.tight_layout(h_pad=1)
        plt.show()