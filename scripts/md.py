import sys
import cace
import torch
from cace.cace.calculators import CACECalculator
from ase.io import read

###### variables ######
# run for outputs of 100pbe_lr 

functional = "pbe"
model_path = f'./lightning_logs/100{functional}_lr/best_model.pth'
DEVICE = 'cuda'
temperature = 300
timestep = 1 #fs #0.25 fs to compute IR spectra
nsteps = 200000
trajectory_file = f'h2o_{functional}_md_1fs_{nsteps}.traj'
logfile = f'h2o_{functional}_md_1fs_{nsteps}.log'

#average_E0 = {1: -5.868579157459375, 8: -2.9342895787296874} #make sure to change this according to the data set you used for training.
# get average energies from remove_mean_cleaned 

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
init_config = read('./liquid-64.xyz', index=0) #can be any starting water config.
atoms = init_config.copy()
atoms.calc = calculator

### Geometry optimization(optional) ###
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
dyn = NPT(atoms, 
                    timestep * units.fs, 
                    temperature_K= temperature,
                    ttime = 10 * timestep * units.fs,
                    pfactor=None,
                    externalstress=0.0
                        ) #NVT setting
##equilibration##
dyn.run(equilibration_steps)
##run##
md_logger = MDLogger(dyn, atoms, logfile=logfile, 
                     header=True, stress=False, mode='w')
dyn.attach(md_logger, interval=100) #md log check
traj = Trajectory(trajectory_file, 'w', atoms)
dyn.attach(traj.write, interval=100) ## will write .traj file
print("Starting equilibration (NVT) ...")
dyn.run(nsteps)
print("complete.")