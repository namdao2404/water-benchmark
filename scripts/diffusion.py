import numpy as np
import matplotlib.pyplot as plt
import sys
from ase.io.trajectory import Trajectory
from scipy import stats
import contextlib
import os

# parameters
timestep = 1.0  # fs
save_interval = 100  # trajectory saved every 100 md steps

if len(sys.argv) < 2:
    raise ValueError("usage: python diffusion.py <functional>")

functional = sys.argv[1]

traj_file = (
    f"/global/scratch/users/namdao2404/"
    f"long_md/long_md_trajectories/1fs_1000ps/"
    f"nam_model_h2o_{functional}_md_1fs_1000ps.traj"
)

# analysis settings
# 50 ps of equilibration already completed before the start of the trajectory
FRAME_SPACING_PS = timestep * save_interval / 1000.0  # 0.1 ps

# analyze every 0.5 ps
ANALYSIS_STRIDE = int(0.5 / FRAME_SPACING_PS)

# fit window for diffusion (and slope diagnostic)
T_MIN_FIT = 100.0  # ps
T_MAX_FIT = None

N_BLOCKS = 5

# fraction of the per block msd length to keep as usable lag
MAX_LAG_FRACTION = 0.8

outdir = (
    "/global/scratch/users/namdao2404/"
    "diffusion/diffusion_1fs_1000ps"
)

os.makedirs(outdir, exist_ok=True)

output_txt = f"{outdir}/diffusion_{functional}.txt"


def msd_fft(pos):
    # fft based msd, borodin algorithm
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


def compute_loglog_slope(time_ps, msd, t_min_ps=10.0, t_max_ps=None):
    mask = (time_ps >= t_min_ps) & (msd > 0)
    if t_max_ps is not None:
        mask &= (time_ps <= t_max_ps)

    t = time_ps[mask]
    y = msd[mask]

    if len(t) < 3:
        return np.nan, np.nan

    result = stats.linregress(np.log(t), np.log(y))

    return result.slope, result.stderr


def calculate_msd_fft(
    traj_path,
    md_timestep,
    save_interval,
    atom_symbol="O",
    start_frame=0,
    end_frame=None,
    stride=1,
    max_lag_fraction=MAX_LAG_FRACTION,
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

    max_lag = int(max_lag_fraction * len(msd))
    return time_ps[:max_lag], msd[:max_lag]


def compute_diffusion_coefficient(time_ps, msd, t_min_ps=1.0, t_max_ps=None):
    mask = time_ps >= t_min_ps
    if t_max_ps is not None:
        mask &= time_ps <= t_max_ps
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


def compute_diffusion_block_averaged(
    traj_path,
    md_timestep,
    save_interval,
    atom_symbol,
    n_blocks=10,
    t_min_ps=1.0,
    t_max_ps=None,
    stride=1,
    max_lag_fraction=MAX_LAG_FRACTION,
):
    # block averaging with propagated uncertainty
    traj = Trajectory(traj_path, "r")
    total_frames = len(traj)

    block_size = total_frames // n_blocks

    D_blocks = []
    D_fit_errors = []

    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < n_blocks - 1 else total_frames

        t, m = calculate_msd_fft(
            traj_path,
            md_timestep,
            save_interval,
            atom_symbol,
            start_frame=start,
            end_frame=end,
            stride=stride,
            max_lag_fraction=max_lag_fraction,
        )

        D_i, D_fit_i, _, _ = compute_diffusion_coefficient(t, m, t_min_ps, t_max_ps)

        if not np.isnan(D_i):
            D_blocks.append(D_i)
            D_fit_errors.append(D_fit_i)

    D_blocks = np.asarray(D_blocks)
    D_fit_errors = np.asarray(D_fit_errors)

    n = len(D_blocks)

    mean_D = np.mean(D_blocks)

    # block to block spread
    block_std = np.std(D_blocks, ddof=1)

    # standard error of the mean
    sem_blocks = block_std / np.sqrt(n)

    total_sem = sem_blocks

    return mean_D, total_sem, block_std, sem_blocks, D_blocks


if __name__ == "__main__":

    with open(output_txt, "w") as f, contextlib.redirect_stdout(f):

        print(f"functional: {functional}")
        print(f"trajectory: {traj_file}")
        print(f"trajectory save interval: {FRAME_SPACING_PS:.3f} ps")
        print(f"analysis stride: {ANALYSIS_STRIDE}")
        print(f"block averaging: {N_BLOCKS} blocks")
        print(f"fit window: {T_MIN_FIT} to {T_MAX_FIT} ps")
        print(f"max lag fraction (per block): {MAX_LAG_FRACTION}")
        print()

        t, msd = calculate_msd_fft(
            traj_file,
            timestep,
            save_interval,
            atom_symbol="O",
            stride=ANALYSIS_STRIDE,
            max_lag_fraction=MAX_LAG_FRACTION,
        )

        D, total_sem, block_std, sem_blocks, D_blocks = compute_diffusion_block_averaged(
            traj_file,
            timestep,
            save_interval,
            atom_symbol="O",
            n_blocks=N_BLOCKS,
            t_min_ps=T_MIN_FIT,
            t_max_ps=T_MAX_FIT,
            stride=ANALYSIS_STRIDE,
            max_lag_fraction=MAX_LAG_FRACTION,
        )

        # msd plot
        plt.figure(figsize=(8, 6))
        plt.plot(t, msd, label="MSD (FFT)")
        plt.axvline(T_MIN_FIT, ls="--", c="k")

        if T_MAX_FIT is not None:
            plt.axvline(T_MAX_FIT, ls="--", c="k")

        plt.xlabel("Time [ps]", fontsize=24)
        plt.ylabel("MSD [Å$^2$]", fontsize=24)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{outdir}/msd_{functional}.png", dpi=300)
        plt.close()

        # log log plot
        plt.figure(figsize=(7, 6))
        fit = (t >= T_MIN_FIT) & (msd > 0)
        if T_MAX_FIT is not None:
            fit &= (t <= T_MAX_FIT)

        plt.loglog(t[(t > 0) & (msd > 0)], msd[(t > 0) & (msd > 0)], label="MSD")

        alpha, alpha_err = compute_loglog_slope(
            t, msd, t_min_ps=T_MIN_FIT, t_max_ps=T_MAX_FIT
        )

        if not np.isnan(alpha):
            coeffs = np.polyfit(np.log(t[fit]), np.log(msd[fit]), 1)
            A = np.exp(coeffs[1])

            t_fit = t[fit]
            msd_fit = A * t_fit**alpha

            plt.loglog(t_fit, msd_fit, '--',
                       label=f"Fit (α = {alpha:.2f} ± {alpha_err:.2f})")

        plt.axvline(T_MIN_FIT, ls="--", c="k")

        if T_MAX_FIT is not None:
            plt.axvline(T_MAX_FIT, ls="--", c="k")

        plt.xlabel("Time [ps]", fontsize=24)
        plt.ylabel("MSD [Å$^2$]", fontsize=24)

        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{outdir}/msd_loglog_{functional}.png", dpi=300)
        plt.close()

        print(f"diffusion coefficient ({N_BLOCKS} block averaged)")
        print(f"D = {D:.6f} ± {total_sem:.6f} Å²/ps")
        print(f"D = {D*1e-16:.2e} ± {total_sem*1e-16:.2e} cm²/s")
        print(f"D = {D*1e-20:.2e} ± {total_sem*1e-20:.2e} m²/s")
        print()
        print(f"log log slope α = {alpha:.3f} ± {alpha_err:.3f} (should be ~1.0)")
        print()
        print(f"block standard deviation (spread): {block_std:.6f}")
        print(f"SEM from blocks only: {sem_blocks:.6f}")
        print(f"blocks used: {len(D_blocks)}")
        print("block values:", D_blocks)

        print(f"msd and diffusion analysis for {functional} complete.")
