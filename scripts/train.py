import os
import glob
import torch
import sys
from collections import OrderedDict
from multihead.data import MultiHeadData
from cace.cace.tasks import LightningData, LightningTrainingTask


#Change data availability & learning rate
# Use the below line when we want all the data for the functionals 
AVAIL_DCT = OrderedDict({"pbe":1})

LEARNING_RATE = 1e-2
#Other vars:
MAX_STEPS = 250000
CUTOFF = 5.5

logs_name = ""
for fnal in AVAIL_DCT:
    pct = int(AVAIL_DCT[fnal]*100)
    if not logs_name:
        logs_name = f"{pct}{fnal}"
    else:
        logs_name += f"_{pct}{fnal}"

on_cluster = False
if 'SLURM_JOB_CPUS_PER_NODE' in os.environ.keys():
    on_cluster = True
if on_cluster:
    #Path tbd
    root = "./remove-mean-cleaned.xyz"
else:
    root = "./remove-mean-cleaned.xyz"


data = MultiHeadData(root=root,batch_size=4,cutoff=CUTOFF,availability_dct=AVAIL_DCT)

#root = "./CsPbBr_3.xyz"

LR = True
if LR:
    logs_name += "_lr"
else:
    logs_name += "_sr"


from cace.cace.representations import Cace
from cace.cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered
from cace.cace.modules import PolynomialCutoff

#Model
radial_basis = BesselRBF(cutoff=CUTOFF, n_rbf=6, trainable=True)
cutoff_fn = PolynomialCutoff(cutoff=CUTOFF)

representation = Cace(
    zs=[1,8],
    n_atom_basis=3,
    embed_receiver_nodes=True,
    cutoff=CUTOFF,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=3,
    max_nu=3,
    num_message_passing=1,
    type_message_passing=['Bchi'],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    avg_num_neighbors=1,
    timeit=False
)

from cace.cace.models import NeuralNetworkPotential, CombinePotential
from cace.cace.modules import Atomwise, Forces, EwaldPotential

atomwise = Atomwise(n_layers=3,
                    output_key="pred_energy",
                    n_hidden=[32,16],
                    n_out=len(AVAIL_DCT),
                    use_batchnorm=False,
                    add_linear_nn=True)

forces = Forces(energy_key="pred_energy",
                forces_key="pred_force")

model = NeuralNetworkPotential(
    input_modules=None,
    representation=representation,
    output_modules=[atomwise,forces]
)

#special LR case taken from DJ's training script
if LR:
    q = Atomwise( #q = cace.modules.Atomwise(
        n_layers=3,
        n_hidden=[24,12],
        n_out=1,
        per_atom_output_key='q',
        output_key = 'tot_q',
        residual=False,
        add_linear_nn=True,
        bias=False)
    
    ep = EwaldPotential(dl=3, #cace.cace.modules.EwaldPotential(dl=3,
                        sigma=1.5,
                        feature_key='q',
                        output_key='ewald_potential',
                        remove_self_interaction=False,
                       aggregation_mode='sum')
    
    forces_lr = Forces(energy_key='ewald_potential',
                                        forces_key='ewald_forces')
    
    cace_nnp_lr = NeuralNetworkPotential(
        input_modules=None,
        representation=representation,
        output_modules=[q, ep, forces_lr]
    )

    pot1 = {'pred_energy': 'pred_energy', 
            'pred_force': 'pred_force',
            'weight': 1,
           }
    
    pot2 = {'pred_energy': 'ewald_potential', 
            'pred_force': 'ewald_forces',
            'weight': 1,
           }
    
    model = CombinePotential([model, cace_nnp_lr], [pot1,pot2])

#Run through a batch to initialize
model.cuda()
for batch in data.val_dataloader():
    model(batch.cuda())
    break

#Important part -- need to feed in all appropriate losses and metrics to track / add
from cace.cace.tasks import LightningTrainingTask
from multihead.data import default_multihead_losses, default_multihead_metrics
losses, metrics = [], []
for i,fnal in enumerate(AVAIL_DCT.keys()):
    #model outputs defined by order of avail_dct
    if len(AVAIL_DCT) == 1:
        losses += default_multihead_losses(fnal,None)
        metrics += default_multihead_metrics(fnal,None)
    else:
        losses += default_multihead_losses(fnal,i)
        metrics += default_multihead_metrics(fnal,i)  

dev_run = False
progress_bar = True
if on_cluster:
    torch.set_float32_matmul_precision('medium')
    progress_bar = False
task = LightningTrainingTask(model,losses=losses,metrics=metrics,save_pkl=True,
                             logs_directory="lightning_logs_4.5",name=logs_name,
                             scheduler_args={'mode': 'min', 'factor': 0.8, 'patience': 10},
                             optimizer_args={'lr': LEARNING_RATE},
                            )
