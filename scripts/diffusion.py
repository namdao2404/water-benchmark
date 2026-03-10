import numpy as np
import matplotlib.pyplot as plt
import sys
from ase.io.trajectory import Trajectory
from scipy import stats
import contextlib
import os

# --------------------------------------------------
# Parameters (unchanged)
# --------------------------------------------------
T = 300
timestep = 1.0        # fs
save_interval = 1
nsteps = 200000

functional = 'pbe'

traj_file = (
    f"/global/scratch/users/namdao2404/md_IR/traj_1fs_files/"
    f"nam_model_h2o_{functional}_5.5_md_h2o_1fs_200000.traj"
)

EQUILIBRATION_STEPS = 20000 #extra equilibration steps
SKIP_FRAMES = EQUILIBRATION_STEPS // save_interval
ANALYSIS_STRIDE = 500 # sample every 0.5ps

outdir = "diffusion_long_equilibration"
os.makedirs(outdir, exist_ok=True)

output_txt = f"{outdir}/diffusion_{functional}.txt"

# --------------------------------------------------
# FFT-based MSD (Borodin algorithm)
# --------------------------------------------------
def msd_fft(pos):
    N, n_atoms, _ = pos.shape

    sq_pos = np.sum(pos**2, axis=2)
    sq_sum = np.sum(sq_pos, axis=1)

    S1 = np.zeros(N)
    Q = 2.0 * np.sum(sq_sum)
    S1[0] = Q
    for m in range(1, N):
        Q -= (sq_sum[m-1] + sq_sum[N-m])
        S1[m] = Q

    S2 = np.zeros(N)
    n_fft = 2**((2*N - 1).bit_length())

    for a in range(n_atoms):
        for d in range(3):
            x = pos[:, a, d]
            X = np.fft.fft(x, n=n_fft)
            ac = np.fft.ifft(X * np.conj(X))[:N].real
            S2 += ac

    denom = np.arange(N, 0, -1)
    return (S1 - 2.0 * S2) / (denom * n_atoms)

def compute_loglog_slope(time_ps, msd, t_min_ps=1.0):

    mask = (time_ps >= t_min_ps) & (msd > 0)

    t = time_ps[mask]
    y = msd[mask]

    if len(t) < 3:
        return np.nan, np.nan

    result = stats.linregress(np.log(t), np.log(y))

    alpha = result.slope
    alpha_err = result.stderr

    return alpha, alpha_err

# --------------------------------------------------
# MSD calculation
# --------------------------------------------------
def calculate_msd_fft(
    traj_path,
    md_timestep,
    save_interval,
    atom_symbol="O",
    start_frame=0,
    end_frame=None,
    stride=1
):
    traj = Trajectory(traj_path, "r")

    if end_frame is None:
        end_frame = len(traj)

    frames = list(range(start_frame, end_frame, stride))
    n_frames = len(frames)

    if n_frames < 2:
        return np.array([]), np.array([])

    atoms0 = traj[frames[0]]
    indices = [i for i, a in enumerate(atoms0) if a.symbol == atom_symbol]

    pos = np.zeros((n_frames, len(indices), 3))

    for i, f in enumerate(frames):
        pos[i] = traj[f].get_positions()[indices]

    msd = msd_fft(pos)

    dt_ps = md_timestep * save_interval * stride / 1000.0
    time_ps = np.arange(len(msd)) * dt_ps

    max_lag = len(msd) // 2
    return time_ps[:max_lag], msd[:max_lag]

# --------------------------------------------------
# Diffusion coefficient + slope uncertainty
# --------------------------------------------------
def compute_diffusion_coefficient(time_ps, msd, t_min_ps=1.0):

    mask = time_ps >= t_min_ps
    t = time_ps[mask]
    y = msd[mask]

    if len(t) < 3:
        return np.nan, np.nan, np.nan, np.nan

    result = stats.linregress(t, y)

    slope = result.slope
    slope_stderr = result.stderr
    r2 = result.rvalue ** 2

    D = slope / 6.0
    D_stderr = slope_stderr / 6.0

    return D, D_stderr, slope, r2

# --------------------------------------------------
# Block averaging with propagated uncertainty
# --------------------------------------------------
def compute_diffusion_block_averaged(
    traj_path,
    md_timestep,
    save_interval,
    atom_symbol,
    n_blocks=8,              # UPDATED TO 8
    t_min_ps=1.0,
    stride=1
):
    traj = Trajectory(traj_path, "r")
    total_frames = len(traj)

    usable_frames = total_frames - SKIP_FRAMES
    block_size = usable_frames // n_blocks

    D_blocks = []
    D_fit_errors = []

    for i in range(n_blocks):
        start = SKIP_FRAMES + i * block_size
        end = SKIP_FRAMES + (i + 1) * block_size if i < n_blocks - 1 else total_frames

        t, m = calculate_msd_fft(
            traj_path,
            md_timestep,
            save_interval,
            atom_symbol,
            start_frame=start,
            end_frame=end,
            stride=stride,
        )

        D_i, D_fit_i, _, _ = compute_diffusion_coefficient(t, m, t_min_ps)

        if not np.isnan(D_i):
            D_blocks.append(D_i)
            D_fit_errors.append(D_fit_i)

    D_blocks = np.asarray(D_blocks)
    D_fit_errors = np.asarray(D_fit_errors)

    n = len(D_blocks)

    mean_D = np.mean(D_blocks)

    # Block-to-block spread
    block_std = np.std(D_blocks, ddof=1)

    # Standard error of mean (descriptive statistics)
    sem_blocks = block_std / np.sqrt(n)

    # Mean regression uncertainty
    # mean_fit_var = np.mean(D_fit_errors**2)
    # ignore this; i was double counting before

    # Propagated SEM (block variance + fit variance)
    # total_sem = np.sqrt(sem_blocks**2 + mean_fit_var)

    total_sem = sem_blocks

    return mean_D, total_sem, block_std, sem_blocks, D_blocks

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    with open(output_txt, "w") as f, contextlib.redirect_stdout(f):

        t, msd = calculate_msd_fft(
            traj_file,
            timestep,
            save_interval,
            atom_symbol="O",
            start_frame=SKIP_FRAMES,
            stride=ANALYSIS_STRIDE,
        )

        D, total_sem, block_std, sem_blocks, D_blocks = compute_diffusion_block_averaged(
            traj_file,
            timestep,
            save_interval,
            atom_symbol="O",
            n_blocks=8,
            t_min_ps=10.0,
            stride=ANALYSIS_STRIDE,
        )

        # ---- MSD plot ----
        plt.figure(figsize=(8, 6))
        plt.plot(t, msd, label="MSD (FFT)")
        plt.axvline(10.0, ls="--", c="k")
        
        plt.xlabel("Time [ps]", fontsize=24)
        plt.ylabel("MSD [Å$^2$]", fontsize=24)
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{outdir}/msd_{functional}.png", dpi=300)
        plt.close()
        
        
        # ---- Log-log plot ----
        plt.figure(figsize=(7, 6))
        fit = (t >= 10.0) & (msd > 0)
        
        plt.figure(figsize=(7, 6))
        plt.loglog(t[fit], msd[fit], label="MSD")
        
        # --- Log–log slope ---
        alpha, alpha_err = compute_loglog_slope(t, msd, t_min_ps=10.0)
        
        if not np.isnan(alpha):
        
            # Power-law fit: MSD = A * t^alpha
            coeffs = np.polyfit(np.log(t[fit]), np.log(msd[fit]), 1)
            A = np.exp(coeffs[1])
        
            t_fit = t[fit]
            msd_fit = A * t_fit**alpha
        
            plt.loglog(t_fit, msd_fit, '--', 
                       label=f"Fit (α = {alpha:.2f} ± {alpha_err:.2f})")
        
        plt.axvline(1.0, ls="--", c="k")
        
        plt.xlabel("Time [ps]", fontsize=24)
        plt.ylabel("MSD [Å$^2$]", fontsize=24)
        
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{outdir}/msd_loglog_{functional}.png", dpi=300)
        plt.close()

        print("=" * 60)
        print("Diffusion coefficient (8-block averaged)")
        print(f"D = {D:.6f} ± {total_sem:.6f} Å²/ps")
        print(f"D = {D*1e-16:.2e} ± {total_sem*1e-16:.2e} cm²/s")
        print(f"D = {D*1e-20:.2e} ± {total_sem*1e-20:.2e} m²/s")
        print()
        print(f"Block standard deviation (spread): {block_std:.6f}")
        print(f"SEM from blocks only: {sem_blocks:.6f}")
        print(f"Blocks used: {len(D_blocks)}")
        print("Block values:", D_blocks)
        print("=" * 60)

        print(f"MSD and diffusion analysis for {functional} complete.")
