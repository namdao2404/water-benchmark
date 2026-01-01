import sys
import cace
import torch
from cace.cace.calculators import CACECalculator
from ase.io import read
import numpy as np 

###### variables ######
# run for outputs of 100pbe_lr 

functional = 'pbe'
model_path = f'./lightning_logs/100{functional}_lr/best_model.pth'
DEVICE = 'cuda'
temperature = 100 # for ice 
timestep = 0.25 # fs; smaller time step for better polarization data 
nsteps = 200000
trajectory_file = f'test_2_nam_model_nvt_ice_h2o_{functional}_5.5_0.25fs_200000.traj'
logfile = f'test_2_nam_model_nvt_ice_h2o_{functional}_5.5_0.25fs_200000.log'


###### load model ######
cace_nnp = torch.load(model_path, map_location=DEVICE, weights_only=False)
#print([module.__class__.__name__ for module in cace_nnp.output_modules])

calculator = CACECalculator(
    model_path=cace_nnp,
    device=DEVICE,
    compute_stress=False,
    energy_key='pred_energy',
    forces_key='pred_force',
    #atomic_energies= average_E0
    atomic_energies=None
)


###### load init_config ######
#init_config = read('/global/scratch/users/namdao2404/ice/struct-ice-Ic-alt3.xyz', index=0) #can be any starting water config.
init_config = read('/global/scratch/users/namdao2404/ice/iceih-64.xyz', index=0)
atoms = init_config.copy()
#atoms.cell = atoms.cell.complete().standard_form()
atoms.wrap()  # make sure atoms are inside the new box
atoms.calc = calculator

# replace with LBFGS for larger systems to optimize geometry more quickly 
from ase.optimize import BFGS
optimizer = BFGS(atoms)
optimizer.run(fmax=0.02)

###### set NVT velocity ######
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
MaxwellBoltzmannDistribution(atoms, temperature_K= temperature )

from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.logger import MDLogger
from ase import Atoms, units
from ase.io.trajectory import Trajectory
equilibration_steps = 5000 ## can choose

print(f"Starting geometry optimization for for ice-phase water for {functional}...")

# Only allow isotropic scaling (all directions scale together)
#mask = np.eye(3)  # 3x3 identity matrix

dyn = NPT(atoms, 
                    timestep * units.fs, 
                    temperature_K= temperature,
                    ttime = 10 * timestep * units.fs,
                    pfactor=None,
                    externalstress=0.0,
                    ) #NVT setting

dyn.run(equilibration_steps)
##run##
md_logger = MDLogger(dyn, atoms, logfile=logfile, 
                     header=True, stress=False, mode='w')
dyn.attach(md_logger, interval=1) #md log check
traj = Trajectory(trajectory_file, 'w', atoms)
dyn.attach(traj.write, interval=1) ## will write .traj file
print(f"Starting equilibration NVT (diffusion coefficient) for ice-phase water for {functional}...")
dyn.run(nsteps)
print("complete.")
