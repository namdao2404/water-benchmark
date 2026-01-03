import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import ase

from ase.io import read
from ase import units
from ase.io.trajectory import Trajectory
# from ase.md.analysis import DiffusionCoefficient # Temporarily comment out for manual approach
from scipy import stats


# --- Parameters (adjust as needed) ---
functional = "pbe"
T = 300  # Temperature (K)
timestep = 1.0   # MD simulation timestep in femtoseconds (fs)
save_interval = 1 # Number of MD steps between saved frames in the trajectory file
nsteps = 200000 # Total number of MD simulation steps

# Path to your trajectory file
traj_file = f'./traj_1fs_files/h2o_{functional}_md_1fs_200000.traj'


# --- Manual MSD Calculation Function ---
def calculate_msd_manually(traj_path, md_timestep, save_interval=1, atom_indices=None):
    """
    Manually compute MSD from trajectory.

    Args:
        traj_path (str): Path to trajectory file (.xyz or .traj format).
        md_timestep (float): MD simulation timestep in fs.
        save_interval (int): Number of MD steps between saved frames in the trajectory file.
        atom_indices (list, optional): List of atom indices for which to calculate MSD. If None, all atoms.

    Returns:
        tuple: (msd_time_ps, msd_values_total)
            msd_time_ps (np.array): Time array for MSD in ps.
            msd_values_total (np.array): Total MSD values in Å².
    """
    # Load trajectory
    try:
        if traj_path.endswith('.xyz'):
            traj = read(traj_path, index=':', format='xyz')
        elif traj_path.endswith('.traj'):
            traj = Trajectory(traj_path, 'r')
        else:
            raise ValueError("Unsupported trajectory file format. Use .xyz or .traj")
        print(f"Successfully loaded {len(traj)} frames from {traj_path}")
    except Exception as e:
        raise ValueError(f"Error reading trajectory file {traj_path}: {e}")

    num_frames = len(traj)  # Correct way to get number of frames in ASE Trajectory

    if len(traj) < 2:
        print(f"Warning: Trajectory has only {len(traj)} frames, need at least 2 for MSD analysis.")
        return np.array([]), np.array([])

    effective_traj_timestep_ps = (md_timestep * save_interval) / 1000.0 # fs to ps

    # Get positions of selected atoms
    all_positions = []
    for atoms in traj:
        if atom_indices is not None:
            # Get indices of oxygen atoms
            oxygen_indices = [i for i, atom in enumerate(atoms) if atom.symbol == atom_indices]
            all_positions.append(atoms.get_positions()[oxygen_indices])
        else:
            all_positions.append(atoms.get_positions())
    all_positions = np.array(all_positions) # Shape: (num_frames, num_selected_atoms, 3)

    num_frames = all_positions.shape[0]
    num_atoms_selected = all_positions.shape[1]

    max_lag_time_index = num_frames // 2 # Common practice: up to half the trajectory length

    msd_values = np.zeros(max_lag_time_index)
    time_points = np.arange(1, max_lag_time_index + 1) * effective_traj_timestep_ps

    # Calculate MSD using multiple time origins
    # This loop can be computationally expensive for very long trajectories
    for lag_time_index in range(1, max_lag_time_index + 1):
        squared_displacements_at_lag = []
        for t0_index in range(num_frames - lag_time_index):
            # Displacement vector for each atom at this time origin and lag time
            displacement_vectors = all_positions[t0_index + lag_time_index] - all_positions[t0_index]
            
            # Squared displacement for each atom
            squared_displacements = np.sum(displacement_vectors**2, axis=1) # Sum over x,y,z components

            # Append to list for averaging
            squared_displacements_at_lag.extend(squared_displacements)
        
        # Mean of squared displacements at this lag time
        if squared_displacements_at_lag: # Ensure list is not empty
            msd_values[lag_time_index - 1] = np.mean(squared_displacements_at_lag)
        else:
            msd_values[lag_time_index - 1] = np.nan # Should not happen if num_frames > 1

    # Filter out NaNs if any (due to short trajectory issues)
    valid_indices = ~np.isnan(msd_values)
    msd_time_ps = time_points[valid_indices]
    msd_values_total = msd_values[valid_indices]

    return msd_time_ps, msd_values_total


def compute_diffusion_coefficient(msd_time_ps, msd_values):
    """
    Compute diffusion coefficient from MSD vs time data using Einstein relation.
    
    Args:
        msd_time_ps (np.array): Time array in ps.
        msd_values (np.array): MSD values in Å².
        
    Returns:
        tuple: (D, slope, r_squared)
            D (float): Diffusion coefficient in Å²/ps.
            slope (float): Slope of the linear fit (which is 6D).
            r_squared (float): R-squared value of the fit.
    """
    if len(msd_time_ps) < 2 or len(msd_values) < 2:
        print("Not enough data points for linear regression.")
        return np.nan, np.nan, np.nan

    # It's common to fit the linear region of MSD. For simplicity, we'll fit all data.
    # For a real analysis, you might want to identify and select the linear regime.
    slope, intercept, r_value, p_value, std_err = stats.linregress(msd_time_ps, msd_values)
    r_squared = r_value ** 2
    
    # Diffusion coefficient D = slope / 6 (Einstein relation for 3D diffusion)
    D = slope / 6.0  # Units: Å²/ps
    
    return D, slope, r_squared

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # Step 1: Calculate MSD manually
        msd_time_ps, msd_values_total = calculate_msd_manually(
            traj_file,
            md_timestep=timestep,
            save_interval=save_interval,
            atom_indices="O" # Consider specific atom types for water if needed, e.g., oxygen atoms
        )

        if len(msd_time_ps) == 0:
            print("MSD calculation failed or returned no data.")
        else:
            # Step 2: Calculate Diffusion Coefficient from MSD
            D, slope, r_squared = compute_diffusion_coefficient(msd_time_ps, msd_values_total)
            D_std = np.nan # Manual calculation doesn't easily provide std from segments

            # --- Plotting MSD vs Time ---
            plt.figure(figsize=(12, 8))
            plt.plot(msd_time_ps, msd_values_total, 'b-', linewidth=1, alpha=0.8, label='MSD Data (Manually Calculated)')

            if not np.isnan(slope):
                plt.plot(msd_time_ps, slope * msd_time_ps + 0, 'r--', linewidth=2,
                         label=f'Linear Fit: $y = {slope:.4f}x$ ($R^2 = {r_squared:.4f}$)')

            plt.xlabel('Time (ps)')
            plt.ylabel('MSD (Å$^2$)')
            plt.title('Mean Squared Displacement (MSD) vs Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

            # --- Print Results ---
            if not np.isnan(D):
                D_cm2_s = D * 1e-16 # Convert Å²/ps to cm²/s
                D_m2_s = D * 1e-20  # Convert Å²/ps to m²/s

                print("\n" + "="*60)
                print("Mean Squared Displacement (MSD) Analysis Results:")
                print(f"Total simulation time analyzed for MSD: {msd_time_ps[-1]:.2f} ps")
                print("="*60)

                print("\n" + "="*60)
                print("Diffusion Coefficient Analysis (Manual MSD Calculation):")
                print(f"Diffusion coefficient (D): {D:.6f} Å²/ps")
                print(f"Diffusion coefficient (D): {D_cm2_s:.2e} cm²/s")
                print(f"Diffusion coefficient (D): {D_m2_s:.2e} m²/s")
                print(f"Linear fit slope (6D): {slope:.6f} Å²/ps")
                print(f"R-squared value of MSD fit: {r_squared:.4f}")
                print(f"Linear fit quality: {'Good' if r_squared > 0.98 else 'Poor'}")
                print("="*60)
            else:
                print("\n" + "="*60)
                print("Diffusion coefficient could not be computed due to insufficient MSD data or poor fit.")
                print("="*60)

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()

    print(f"MSD and diffusion analysis for {functional} complete.")
