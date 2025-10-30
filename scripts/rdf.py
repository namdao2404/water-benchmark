import sys
import numpy as np
import torch
from ase import Atoms
from ase.geometry import get_distances

class idxFormatError(Exception):
    pass

def rdf(images, idx1, idx2, exclude_idx=None, binwidth=0.1, surface=False, molecule=None):
    """
    """
    if isinstance(images, Atoms):
        images = [images]
    else:
        images = list(images)
    atoms0 = images[0]

    if atoms0.get_volume() == 0:
        pos0 = atoms0.get_positions()
        atoms0.set_positions(pos0 - np.min(pos0, axis=0))
        atoms0.cell = np.max(atoms0.get_positions(), axis=0)
    vol = atoms0.get_volume()

    excl = [] if exclude_idx is None else ([exclude_idx] if isinstance(exclude_idx, int) else list(exclude_idx))
    if not all(isinstance(i, int) for i in excl):
        raise idxFormatError("exclude_idx int int .")

    def parse_idx(atoms, idx):
        if isinstance(idx, (int, str)):
            lst = [idx]
        else:
            lst = list(idx)
        syms = atoms.get_chemical_symbols()
        out = []
        for x in lst:
            if isinstance(x, int):
                out.append(x)
            elif isinstance(x, str):
                if x not in set(syms):
                    raise idxFormatError(f" '{x}'")
                out += [i for i,s in enumerate(syms) if s == x]
            else:
                raise idxFormatError("idx int, str, .")
        return sorted(set(out) - set(excl))

    idx1_arr = np.array(parse_idx(atoms0, idx1), dtype=int)
    idx2_arr = np.array(parse_idx(atoms0, idx2), dtype=int)
    if idx1_arr.size==0 or idx2_arr.size==0:
        raise idxFormatError("idx1  idx2.")

    if surface:
        maxz = np.max(atoms0.get_positions()[:,2])
        r_max = maxz + 10.0
    else:
        r_max = np.max(atoms0.cell.diagonal())/2.0
    nbins = int(np.ceil(r_max/binwidth))
    edges = np.linspace(0.0, r_max, nbins+1)
    dr = binwidth
    counts = np.zeros(nbins, float)

    I1, I2 = np.meshgrid(idx1_arr, idx2_arr, indexing='ij')
    p1, p2 = I1.ravel(), I2.ravel()
    if molecule is not None:
        mask = (p1!=p2) & ((p1//molecule)!=(p2//molecule))
    else:
        mask = (p1!=p2)

    for atoms in images:
        pos = atoms.get_positions()
        if surface:
            d = np.abs(pos[idx1_arr,2][:,None] - pos[idx2_arr,2][None,:]).ravel()
        else:
            _, dmat = get_distances(pos[idx1_arr], pos[idx2_arr],
                                    cell=atoms.cell, pbc=atoms.pbc)
            d = dmat.ravel()
        d = d[mask]
        h, _ = np.histogram(d, bins=edges)
        counts += h

    N_i   = idx1_arr.size        
    rho_j = idx2_arr.size / vol 
    n_img = len(images)

    rs = 0.5*(edges[:-1] + edges[1:])
    g_r = []
    for i, r in enumerate(rs):
        shell_vol = 4.0 * np.pi * r*r * dr
        norm = N_i * n_img * shell_vol * rho_j
        g_r.append((r, counts[i] / norm))

    return g_r


from ase.io import read
functional = "pbe"
traj = read(f'./traj_1fs_files/h2o_{functional}_5.5_md_h2o_1fs_200000.traj', index=':')
g_r_oo_1 = rdf(traj, "O", "O", binwidth=0.05)
g_r_oh_1 = rdf(traj, "O", "H", binwidth=0.05)
g_r_hh_1 = rdf(traj, "H", "H", binwidth=0.05)

x_oo_1, y_oo_1 = zip(*g_r_oo_1)
x_oh_1, y_oh_1 = zip(*g_r_oh_1)
x_hh_1, y_hh_1 = zip(*g_r_hh_1)
r   = np.array(x_oo_1)
gOO = np.array(y_oo_1)
gOH = np.array(y_oh_1)
gHH = np.array(y_hh_1)

data = np.column_stack((r, gOO, gOH, gHH))


import os
np.savetxt(
    f'{functional}_rdf_water.csv',
    data,
    delimiter=',',
    header='r,g_OO,g_OH,g_HH',
    comments=''
)