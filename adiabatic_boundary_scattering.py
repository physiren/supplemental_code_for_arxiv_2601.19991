#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Floquet band structure and transmission spectrum calculation
- Computes Floquet quasienergy bands with group velocity information
- Calculates transmission spectrum for comparison
- Follows physrev style guidelines for figures
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba as nb
from multiprocessing import Pool
import os
import time
from matplotlib import rcParams
import potentials
import lib

# Set physrev style parameters
plt.style.use('default')
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 12
rcParams['legend.fontsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['figure.autolayout'] = True

#-----------------------------------
# Global safety parameters
#-----------------------------------
G_SAFE = 8       # Momentum cutoff safety factor
OMEGA_SAFE = 16   # Time sampling safety factor

#-----------------------------------
# Experimental parameter settings (single set of parameters)
#-----------------------------------
EXPERIMENT_PARAMS = {
    'shift': 2,    # Energy shift
    'V1': 0,       # Strength of first potential term
    'V2': 8,       # Strength of second potential term
    'n1': 0,       # Integer for first term: cos(2πx - n1 ωt)
    'n2': 1,       # Integer for second term: cos(2πx - n2 ωt)
    'omega': 2,    # Driving frequency
    'N_k': 500,    # Number of k-space sampling points
    'auto_cutoff': True  # Automatically calculate cutoff parameters
}

x_mesh = 64  # Spatial mesh points

#-----------------------------------
# Automatic parameter calculation function
#-----------------------------------
def calculate_cutoffs(params):
    """Automatically calculate cutoff parameters"""
    V1 = params['V1']
    V2 = params['V2']
    omega = params['omega']
    
    # Calculate momentum cutoff
    if params['auto_cutoff']:
        G_max = np.sqrt(G_SAFE * (V1 + V2))
        N_side = int(np.floor(G_max / (2 * np.pi))) + 1
        params['N_basis'] = 2 * N_side + 1
        
        # Calculate number of time steps (nearest power of 2)
        M_raw = (V1 + V2) * G_SAFE * OMEGA_SAFE / omega
        M = int(2**np.ceil(np.log2(M_raw)))
        params['M'] = M
        
    return params

#-----------------------------------
# Hamiltonian matrix construction accelerated with Numba
#-----------------------------------
@nb.njit
def H_matrix_numba(k, t, G_range, V1, V2, shift, n1, n2, omega):
    """Numba-accelerated Hamiltonian matrix construction"""
    N = len(G_range)
    H = np.zeros((N, N), dtype=nb.complex128)
    
    # Kinetic energy term
    for i in range(N):
        H[i, i] = (k + G_range[i])**2 + shift
    
    # Potential energy term
    for i in range(N - 1):
        coupling = 0.5 * (V1 * np.exp(-1j * n1 * omega * t) + V2 * np.exp(-1j * n2 * omega * t))
        H[i + 1, i] = coupling
        H[i, i + 1] = np.conjugate(coupling)
    
    return H

#-----------------------------------
# Time evolution step accelerated with Numba
#-----------------------------------
@nb.njit
def evolve_step_numba(H, dt):
    """Single time evolution step, accelerated with Numba"""
    eigval, eigvec = np.linalg.eigh(H)
    exp_diag = np.exp(-1j * eigval * dt)
    return eigvec @ np.diag(exp_diag) @ eigvec.conj().T

#-----------------------------------
# Time evolution operator calculation
#-----------------------------------
def time_evolve_operator(k, params):
    """Calculate time evolution operator for given k"""
    # Extract required values from parameter dictionary
    V1 = params['V1']
    V2 = params['V2']
    shift = params['shift']
    n1 = params['n1']
    n2 = params['n2']
    omega = params['omega']
    M = params['M']
    
    # Calculate period and time step
    T = 2 * np.pi / omega
    dt = T / M
    
    # Calculate G_range
    N_basis = params['N_basis']
    G_range = np.arange(-N_basis // 2, N_basis // 2 + 1) * 2 * np.pi
    
    N = len(G_range)
    U_total = np.eye(N, dtype=complex)
    U_list = []
    
    for j in range(M):
        t = (j + 1) * dt
        H_t = H_matrix_numba(k, t, G_range, V1, V2, shift, n1, n2, omega)
        U_step = evolve_step_numba(H_t, dt)
        U_total = U_step @ U_total
        U_list.append(U_total.copy())
    
    return U_total, U_list, T, dt

#-----------------------------------
# Group velocity calculation function
#-----------------------------------
def calculate_group_velocity(k, epsilon, params):
    """
    Calculate group velocity for given k value and quasienergy
    
    Parameters:
    - k: Wave vector value
    - epsilon: Quasienergy at this k value
    - params: Parameter dictionary
    
    Returns:
    - v_g: Group velocity v_g = dε/dk
    """
    eps = 1e-6  # Finite difference step
    omega = params['omega']
    
    # Calculate evolution operator and quasienergy at k+eps
    U_F_plus, _, T, _ = time_evolve_operator(k + eps, params)
    eigvals_plus, _ = np.linalg.eig(U_F_plus)
    eps_plus = -np.angle(eigvals_plus) / T
    eps_folded_plus = (eps_plus + omega / 2) % omega - omega / 2
    
    # Find quasienergy value closest to the original one
    idx = np.argmin(np.abs(eps_folded_plus - epsilon))
    epsilon_plus = eps_folded_plus[idx]
    
    # Handle periodic boundaries
    diff = epsilon_plus - epsilon
    if diff > omega / 2:
        diff -= omega
    elif diff < -omega / 2:
        diff += omega
    
    # Calculate group velocity
    v_g = diff / eps
    
    return v_g

#-----------------------------------
# Function to process a single k-point
#-----------------------------------
def process_k_point_with_params(args):
    """Process calculation for a single k-point, return result list"""
    k, params = args
    results = []
    
    # Calculate time evolution operator
    U_F, U_list, T, dt = time_evolve_operator(k, params)
    M = params['M']
    omega = params['omega']
    
    # Calculate quasienergy and eigenstates
    eigvals, eigvecs = np.linalg.eig(U_F)
    eps = -np.angle(eigvals) / T
    eps_folded = (eps + omega / 2) % omega - omega / 2

    # Process each eigenstate
    for n in range(len(eps_folded)):
        v0 = eigvecs[:, n]
        epsilon = eps_folded[n]
        
        # Calculate group velocity
        v_g = calculate_group_velocity(k, epsilon, params)
        
        psi_time = np.array([U @ v0 for U in U_list])
        t_array = np.array([(j + 1) * dt for j in range(M)])
        u_time = psi_time * np.exp(1j * epsilon * t_array)[:, None]
        
        ft = np.fft.fft(u_time, axis=0)
        ft = np.fft.fftshift(ft, axes=0)
        amplitudes = np.linalg.norm(ft, axis=1)
        weight = np.abs(amplitudes)**2
        weight /= np.sum(weight)
        m_vals = np.arange(-M // 2, M // 2)

        for idx, m in enumerate(m_vals):
            if weight[idx] > 1e-3:  # Only save points with sufficient weight
                E_ext = epsilon - m * omega
                results.append((k / (2 * np.pi), E_ext, weight[idx], v_g))
    
    return results

#-----------------------------------
# Main calculation function
#-----------------------------------
def main_calculation(params):
    """Main calculation function"""
    # Calculate automatic cutoff parameters
    params = calculate_cutoffs(params)
    
    print("\nCalculation parameters:")
    print(f"V1={params['V1']}, V2={params['V2']}, ω={params['omega']}")
    print(f"Auto parameters: N_basis={params['N_basis']}, M={params['M']}")
    
    start_time = time.time()
    
    # Prepare k-points
    k_list = np.linspace(-0.5, 0.5, params['N_k']) * 2 * np.pi
    
    # Create parameter tuples for each k-point
    param_list = [(k, params.copy()) for k in k_list]
    
    # Process all k-points in parallel
    num_processes = os.cpu_count() - 1 or 1
    with Pool(processes=num_processes) as pool:
        all_results = list(tqdm(
            pool.imap(process_k_point_with_params, param_list), 
            total=len(k_list), 
            desc="Processing k values"
        ))
    
    # Merge results
    flat_results = [item for sublist in all_results for item in sublist]
    
    if flat_results:
        print(f"\nCalculation completed in {time.time() - start_time:.2f} seconds")
        print(f"Total data points: {len(flat_results)}")
        
        # Return results
        return flat_results, params
    else:
        print("No valid data points found!")

#-----------------------------------
# Transmission spectrum calculation function
#-----------------------------------
def calculate_transmission_spectrum(params, E_mesh=16, frqcutoff_oneside=16, N_adiabatic=0):
    """Calculate transmission spectrum"""
    V1 = params['V1']
    V2 = params['V2']
    omega = params['omega']
    shift = params['shift']
    
    # Energy grid for one period [-ω/2, ω/2]
    E0_list = np.linspace(-omega / 2 + omega / E_mesh / 2, 
                         omega / 2 - omega / E_mesh / 2, 
                         E_mesh, endpoint=True)
    print(E0_list)

    # Extended energy grid for all frequency components
    E_list = np.linspace(-omega / 2 + omega / E_mesh / 2 - frqcutoff_oneside * omega,
                         omega / 2 - omega / E_mesh / 2 + frqcutoff_oneside * omega,
                         E_mesh * (2 * frqcutoff_oneside + 1), endpoint=True)

    # Parameter settings
    mu = 0.5
    nd1 = 0
    nd2 = 1
    d = 1
    N = 2 * frqcutoff_oneside + 1
    
    # Preallocate results arrays
    RL_list = np.zeros(E_mesh * N)
    RR_list = np.zeros(E_mesh * N)
    TL_list = np.zeros(E_mesh * N)
    TR_list = np.zeros(E_mesh * N)
    TL_0_list = np.zeros(E_mesh * N)
    TR_0_list = np.zeros(E_mesh * N)

    # Compute for all energy points in E0_list
    for i, E in enumerate(E0_list):
        parameter_M = potentials.double_wave(mu, V1, V2, shift, nd1, nd2, d, omega, 
                                          x_mesh, frqcutoff_oneside, E)

        TrsfL, TrsfR = lib.transfer_matrix(parameter_M, side='both')

        # Create adiabatic waveguide
        # Initialize S-matrix 4 blocks
        TL_left = np.eye(N, dtype=np.complex128)
        TR_left = np.eye(N, dtype=np.complex128)
        RL_left = np.zeros((N, N), dtype=np.complex128)
        RR_left = np.zeros((N, N), dtype=np.complex128)

        TL_right = np.eye(N, dtype=np.complex128)
        TR_right = np.eye(N, dtype=np.complex128)
        RL_right = np.zeros((N, N), dtype=np.complex128)
        RR_right = np.zeros((N, N), dtype=np.complex128)
        
        # Create F_space
        F_space = lib.gen_free_space(frqcutoff_oneside, E, omega)

        parameter_M = potentials.double_wave(mu, V1, V2, shift, nd1, nd2, d, omega, 
                                    x_mesh, frqcutoff_oneside, E)
        TrsfL, TrsfR = lib.transfer_matrix(parameter_M, side='both')
        eigvals, eigvecs, NLedge_S = lib.diagonalize_TL_TR(N, TrsfL, TrsfR)
        S_space = lib.gen_space(eigvecs, NLedge_S)

        # Recursively calculate S-matrix for adiabatic connection region
        for conductor_index in range(N_adiabatic):
            factor = (conductor_index + 1) / (N_adiabatic + 1)
            parameter_M = potentials.double_wave(mu, factor * V1, factor * V2, factor * shift, 
                                                nd1, nd2, d, omega, x_mesh, frqcutoff_oneside, E)
            TrsfL, TrsfR = lib.transfer_matrix(parameter_M, side='both')
            eigvals, eigvecs, NLedge = lib.diagonalize_TL_TR(N, TrsfL, TrsfR)
            space = lib.gen_space(eigvecs, NLedge)
            D = (eigvals[0:N], eigvals[N:2 * N])
            TL_next, RL_next = lib.tri_region_exact(F_space, space, D, 1, F_space, side='left')
            TR_next, RR_next = lib.tri_region_exact(F_space, space, D, 1, F_space, side='right')

            TL_left, TR_left, RL_left, RR_left = lib.S_matrix_merge(TL_left, TR_left, RL_left, RR_left, 
                                                                    TL_next, TR_next, RL_next, RR_next)

            TL_right, TR_right, RL_right, RR_right = lib.S_matrix_merge(TL_next, TR_next, RL_next, RR_next, 
                                                                        TL_right, TR_right, RL_right, RR_right)

        TL_left_sinf, RL_left_sinf = lib.semi_inf(F_space, S_space, 'left')
        TR_left_sinf, RR_left_sinf = lib.semi_inf(F_space, S_space, 'right')

        NLedge = NLedge_S

        invterm = np.linalg.inv(np.eye(N) - RR_left @ RL_left_sinf)
        R1 = RL_left + TR_left @ RL_left_sinf @ invterm @ TL_left
        T1 = TL_left_sinf @ invterm @ TL_left

        invterm = np.linalg.inv(np.eye(N) - RL_left_sinf @ RR_left)
        T2 = TR_left @ invterm @ TR_left_sinf
        R2 = RR_left_sinf + TL_left_sinf @ RR_left @ invterm @ TR_left_sinf

        TL_right_sinf, RL_right_sinf = lib.semi_inf(S_space, F_space, 'left')
        TR_right_sinf, RR_right_sinf = lib.semi_inf(S_space, F_space, 'right')

        invterm = np.linalg.inv(np.eye(N) - RR_right_sinf @ RL_right)
        T3 = TL_right @ invterm @ TL_right_sinf
        R3 = RL_right_sinf + TR_right_sinf @ RL_right @ invterm @ TL_right_sinf

        invterm = np.linalg.inv(np.eye(N) - RL_right @ RR_right_sinf)
        T4 = TR_right_sinf @ invterm @ TR_right
        R4 = RR_right + TL_right @ RR_right_sinf @ invterm @ TR_right

        Tsqr_left, Rsqr_left = lib.average_trans(20000, R1, R2, R3, T1, T2, T3, NLedge, N)
        Tsqr_right, Rsqr_right = lib.average_trans(20000, R4, R3, R2, T4, T3, T2, NLedge, N)

        metric = np.linspace(-omega * frqcutoff_oneside, omega * frqcutoff_oneside, N, endpoint=True) + E
        metric = (np.sign(metric) > 0).astype(int)
        
        tl_results = np.zeros(N)
        tr_results = np.zeros(N)
        tl_0_results = np.zeros(N)
        tr_0_results = np.zeros(N)

        for j in range(N):
            tl_results[j] = np.dot(Tsqr_left[:, j], metric)
            tr_results[j] = np.dot(Tsqr_right[:, j], metric)
            tl_0_results[j] = Tsqr_left[j, j]
            tr_0_results[j] = Tsqr_right[j, j]

        TL_list[i::E_mesh] = tl_results
        TR_list[i::E_mesh] = tr_results
        TL_0_list[i::E_mesh] = tl_0_results
        TR_0_list[i::E_mesh] = tr_0_results

    return E_list, RL_list, RR_list, TL_list, TR_list, TL_0_list, TR_0_list


if __name__ == "__main__":
    # Use preset experimental parameters for calculation
    results, params = main_calculation(EXPERIMENT_PARAMS)
    
    # Extract plotting data
    k_points = np.array([r[0] for r in results])
    energies = np.array([r[1] for r in results])
    weights = np.array([r[2] for r in results])
    velocities = np.array([r[3] for r in results])
    
    V1 = params['V1']
    V2 = params['V2']
    omega = params['omega']
    
    # Automatically determine energy range
    E_min = 0
    E_max = 20
    
    # Define N_adiabatic values to calculate (only 0, 4, 16)
    N_adiabatic_values = [0, 4, 16]
    transmission_results = []
    
    # Calculate transmission spectrum for each N_adiabatic value
    for N_adiabatic in N_adiabatic_values:
        print(f"\nCalculating transmission spectrum for N_adiabatic = {N_adiabatic}")
        E_list, RL_list, RR_list, TL_list, TR_list, TL_0_list, TR_0_list = calculate_transmission_spectrum(
            params, N_adiabatic=N_adiabatic
        )
        transmission_results.append((N_adiabatic, E_list, TL_list, TR_list, TL_0_list, TR_0_list))
    
    # Create figure conforming to physrev style
    fig = plt.figure(figsize=(8, 4))  # Adjust width to accommodate three transmission spectra
    
    # Create subplot layout: left band diagram, right three transmission spectra horizontally arranged
    n_plots = 1 + len(N_adiabatic_values)  # Band diagram + 3 transmission spectra
    gs = fig.add_gridspec(1, n_plots, width_ratios=[2] + [1] * len(N_adiabatic_values), wspace=0.0)
    
    # Create subplot list
    ax_band = fig.add_subplot(gs[0])  # Band diagram
    ax_trans_list = [fig.add_subplot(gs[i + 1], sharey=ax_band) for i in range(len(N_adiabatic_values))]
    
    # Plot band diagram
    # Assign color to each point based on group velocity
    colors = []
    for v_g, w in zip(velocities, weights):
        if v_g > 0:  # Positive group velocity, blue
            colors.append((0, 0, 1, w))
        else:  # Negative group velocity, red
            colors.append((1, 0, 0, w))
    
    ax_band.scatter(k_points, energies, color=colors, s=3, edgecolors='none')
    
    # Set band diagram axis labels
    ax_band.set_xlabel("Quasimomentum $k/(2\pi)$")
    ax_band.set_ylabel("Extended Quasienergy / Incident Energy $E$")
    
    # Set band diagram axis limits
    ax_band.set_ylim(E_min, E_max)
    ax_band.set_xlim(-0.5, 0.5)  # First Brillouin zone
    
    # Add horizontal dashed lines y = ω/2 + nω
    n_min = int(np.floor((E_min - omega / 2) / omega))
    n_max = int(np.ceil((E_max - omega / 2) / omega))
    for n in range(n_min, n_max + 1):
        y_line = omega / 2 + n * omega
        if E_min <= y_line <= E_max:
            ax_band.axhline(y=y_line, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Add legend explaining color meaning
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, 
              label='$v_g > 0$'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, 
              label='$v_g < 0$')
    ]
    ax_band.legend(handles=legend_elements, loc='upper right', frameon=False)
    
    # Create legend labels (create only once)
    lines = [
        plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=1, alpha=0.5),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1, alpha=0.5),
        plt.Line2D([0], [0], color='blue', linewidth=1, alpha=0.5),
        plt.Line2D([0], [0], color='red', linewidth=1, alpha=0.5)
    ]
    labels = ['          ', '          ', '          ', '          ']
    
    # Plot multiple transmission spectrum subplots (horizontally arranged)
    for i, (ax, (N_adiabatic, E_list, TL_list, TR_list, TL_0_list, TR_0_list)) in enumerate(zip(ax_trans_list, transmission_results)):
        # Plot transmission spectrum
        ax.plot(TL_list, E_list, color='blue', linewidth=1, linestyle='--', alpha=0.5)
        ax.plot(TR_list, E_list, color='red', linewidth=1, linestyle='--', alpha=0.5)
        ax.plot(TL_0_list, E_list, color='blue', linewidth=1, alpha=0.5)
        ax.plot(TR_0_list, E_list, color='red', linewidth=1, alpha=0.5)
        
        # Set axis limits
        ax.set_ylim(E_min, E_max)
        ax.set_xlim(0, 1)

        n_min = int(np.floor((E_min - omega / 2) / omega))
        n_max = int(np.ceil((E_max - omega / 2) / omega))
        for n in range(n_min, n_max + 1):
            y_line = omega / 2 + n * omega
            if E_min <= y_line <= E_max:
                ax.axhline(y=y_line, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Add N value label in top left corner
        connectxt = 'grad' 
        ax.text(0.05, 0.95, fr'$N_{{{connectxt}}} = {N_adiabatic}$',
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Only show x-axis title and values for middle N=4 subplot
        if i == 1:  # N=4 is the middle subplot
            ax.set_xlabel("Current Transmission")
            # Add x-axis ticks
            ax.set_xticks([0, 0.5, 1.0])
        else:
            # Remove x-axis labels and ticks
            ax.set_xlabel("")
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_xticklabels([])
        
        # Ensure all subplots show y-axis ticks
        ax.tick_params(axis='y', which='both', labelleft=False)
    
    # Add legend to last subplot (N=16)
    ax_trans_list[-1].legend(lines, labels, frameon=False, fontsize=8, loc='center right')
    
    # Adjust layout - ensure all subplots have enough space to display y-axis labels
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.92, wspace=0.05)
    
    # Save figure
    plt.savefig('floquet_band_structure_with_transmissions.pdf', format='pdf', bbox_inches='tight')
    plt.show()