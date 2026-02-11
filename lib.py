"""
This script contains various solver functions. To maximize code flexibility, this script does not contain
inputs related to specific problems. All problem-related information is used elsewhere to compute the M matrix list.
This script only accepts the M matrix list and other necessary inputs.
"""

import numpy as np
import scipy as sp
from numba import jit, NumbaPerformanceWarning
from matplotlib import pyplot as plt
import warnings
from julia.api import Julia
import potentials

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# Flag variable to ensure Julia environment and functions are initialized only once
_julia_initialized = False

def initialize_julia():
    """
    Initialize Julia environment for high-precision matrix inversion.
    This function is called automatically when needed and only runs once.
    """
    global _julia_initialized
    if not _julia_initialized:
        # Initialize Julia environment
        jl = Julia(compiled_modules=False)

        # Define a Julia function to handle the entire computation
        jl.eval("""
        using LinearAlgebra

        function compute_R_T(A1, A2, B1, B2, C1, C2)
            # Dynamic precision matrix inversion function
            function safe_inv(matrix)
                cond_num = cond(matrix)
                println("Condition number: ", cond_num)
                
                extra_bits = 10  # Additional safety bits
                precision_bits = max(64, Int(ceil(log2(cond_num))) + extra_bits)
                
                println("Using precision: ", precision_bits, " bits")
                
                if precision_bits > 64
                    setprecision(precision_bits) do
                        big_matrix = Complex{BigFloat}.(matrix)  # Convert to arbitrary precision complex
                        ComplexF64.(inv(big_matrix))             # Convert back to double precision complex
                    end
                else
                    inv(matrix)  # Use double precision directly
                end
            end

            # Compute inverse matrices
            B1i = safe_inv(B1)
            B2i = safe_inv(B2)
            C1i = safe_inv(C1)
            C2i = safe_inv(C2)

            # Compute T and R
            T = inv(B1i * C1 - B2i * C2) * (B1i * A1 - B2i * A2)
            R = -inv(C1i * B1 - C2i * B2) * (C1i * A1 - C2i * A2)

            return R, T
        end
        """)
        _julia_initialized = True


@jit(nopython=True)
def eigh_nb(A):
    """Hermitian eigenvalue decomposition using Numba acceleration"""
    return np.linalg.eigh(A)


@jit(nopython=True)
def eig_nb(A):
    """General eigenvalue decomposition using Numba acceleration"""
    print('start eig')
    result = np.linalg.eig(A)
    print('finished eig')
    return result


@jit(nopython=True)
def dot(A, B):
    """Matrix multiplication using Numba acceleration"""
    return A @ B


@jit(nopython=True)
def positive_root(N, D):
    """
    Compute square root for real numbers:
    - If non-negative, output sqrt(x)
    - If negative, output i*sqrt(-x)
    
    Parameters:
    -----------
    N : int
        Array size
    D : ndarray
        Input array
        
    Returns:
    --------
    result : complex ndarray
        Array with appropriate square roots
    """
    result = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        if D[i] >= 0:
            result[i] = np.sqrt(D[i])
        else:
            result[i] = 1j * np.sqrt(-D[i])
    return result


@jit(nopython=True)
def four_matrix(N, D_M, dx):
    """
    Intermediate step for analytically diagonalizing M matrix and computing infinitesimal symplectic matrix.
    
    Parameters:
    -----------
    N : int
        Frequency truncation number
    D_M : ndarray
        Eigenvalues of M matrix
    dx : float
        Spatial step size
        
    Returns:
    --------
    D11, D12, D21, D22 : diagonal matrices
        Four diagonal matrices for symplectic propagation
    """
    D = positive_root(N, D_M)
    D11 = np.zeros(N, dtype=np.complex128)
    D12 = np.zeros(N, dtype=np.complex128)
    D21 = np.zeros(N, dtype=np.complex128)
    D22 = np.zeros(N, dtype=np.complex128)
    
    for i in range(N):
        e1 = np.exp(D[i] * dx)
        e2 = 1 / e1
        D11[i] = e1 + e2
        
        if abs(D[i]) > 1e-16:
            D12[i] = (e1 - e2) / D[i]
        else:
            print('too small eigenvalue, using Taylor expansion')
            D12[i] = 2 * dx + (D[i]**2 * dx**3) / 3 + (D[i]**4 * dx**5) / 60
            
        D21[i] = (e1 - e2) * D[i]
        D22[i] = D11[i]
    
    return np.diag(D11), np.diag(D12), np.diag(D21), np.diag(D22)


def kernel_TM_np_exp(x_mesh, dx, N, direct, M_list, Efiled=False):
    """
    Compute transfer matrix using exponential method with NumPy.
    
    Parameters:
    -----------
    x_mesh : int
        Number of spatial mesh points
    dx : float
        Spatial step size
    N : int
        Total frequency truncation number
    direct : int
        Direction flag: 0=left, 1=right, 2=both
    M_list : ndarray
        3D array storing M matrices (first dimension is x, last two are matrix dimensions)
    Efiled : bool
        Whether electric field is present
        
    Returns:
    --------
    T or (TL, TR) : transfer matrix/matrices
    """
    infsimal = np.zeros((2 * N, 2 * N), dtype=np.complex128)
    
    if direct == 0 or direct == 1:
        T = np.eye(2 * N)
        for index in range(1, x_mesh):
            if direct == 0:
                i = index
            elif direct == 1:
                i = x_mesh - index
                dx = -dx
                
            D_M, U = eigh_nb(M_list[i, :, :])
            Udag = U.conjugate().T
            D11, D12, D21, D22 = four_matrix(N, D_M, dx)
            
            infsimal[0:N, 0:N] = 0.5 * U @ D11 @ Udag
            infsimal[0:N, N:2*N] = 0.5 * U @ D12 @ Udag
            infsimal[N:2*N, 0:N] = 0.5 * U @ D21 @ Udag
            infsimal[N:2*N, N:2*N] = 0.5 * U @ D22 @ Udag
            T = infsimal @ T
            
        D_M, U = eigh_nb(M_list[0, :, :])
        Udag = U.conjugate().T
        D11, D12, D21, D22 = four_matrix(N, D_M, 0.5 * dx)
        
        infsimal[0:N, 0:N] = 0.5 * U @ D11 @ Udag
        infsimal[0:N, N:2*N] = 0.5 * U @ D12 @ Udag
        infsimal[N:2*N, 0:N] = 0.5 * U @ D21 @ Udag
        infsimal[N:2*N, N:2*N] = 0.5 * U @ D22 @ Udag
        T = infsimal @ T @ infsimal
        return T

    elif direct == 2:
        TL = np.eye(2 * N)
        TR = np.eye(2 * N)
        
        for i in range(1, x_mesh):
            D_M, U = eigh_nb(M_list[i, :, :])
            Udag = U.conjugate().T
            D11, D12, D21, D22 = four_matrix(N, D_M, dx)
            
            infsimal[0:N, 0:N] = 0.5 * U @ D11 @ Udag
            infsimal[0:N, N:2*N] = 0.5 * U @ D12 @ Udag
            infsimal[N:2*N, 0:N] = 0.5 * U @ D21 @ Udag
            infsimal[N:2*N, N:2*N] = 0.5 * U @ D22 @ Udag
            TL = infsimal @ TL
            
            D11, D12, D21, D22 = four_matrix(N, D_M, -dx)
            infsimal[0:N, 0:N] = 0.5 * U @ D11 @ Udag
            infsimal[0:N, N:2*N] = 0.5 * U @ D12 @ Udag
            infsimal[N:2*N, 0:N] = 0.5 * U @ D21 @ Udag
            infsimal[N:2*N, N:2*N] = 0.5 * U @ D22 @ Udag
            TR = TR @ infsimal
        
        D_M, U = eigh_nb(M_list[0, :, :])
        Udag = U.conjugate().T
        
        if Efiled == True:  
            D_M_f, U_f = eigh_nb(M_list[x_mesh, :, :])
            Udag_f = U_f.conjugate().T
            
        D11, D12, D21, D22 = four_matrix(N, D_M, 0.5 * dx)
        infsimal[0:N, 0:N] = 0.5 * U @ D11 @ Udag
        infsimal[0:N, N:2*N] = 0.5 * U @ D12 @ Udag
        infsimal[N:2*N, 0:N] = 0.5 * U @ D21 @ Udag
        infsimal[N:2*N, N:2*N] = 0.5 * U @ D22 @ Udag
        TL = TL @ infsimal
        
        if Efiled == False:
            TL = infsimal @ TL
        else:
            D11, D12, D21, D22 = four_matrix(N, D_M_f, 0.5 * dx)
            infsimal[0:N, 0:N] = 0.5 * U @ D11 @ Udag_f
            infsimal[0:N, N:2*N] = 0.5 * U @ D12 @ Udag_f
            infsimal[N:2*N, 0:N] = 0.5 * U @ D21 @ Udag_f
            infsimal[N:2*N, N:2*N] = 0.5 * U @ D22 @ Udag_f
            TL = infsimal @ TL

        D11, D12, D21, D22 = four_matrix(N, D_M, -0.5 * dx)
        infsimal[0:N, 0:N] = 0.5 * U @ D11 @ Udag
        infsimal[0:N, N:2*N] = 0.5 * U @ D12 @ Udag
        infsimal[N:2*N, 0:N] = 0.5 * U @ D21 @ Udag
        infsimal[N:2*N, N:2*N] = 0.5 * U @ D22 @ Udag
        TR = infsimal @ TR
        
        if Efiled == False:
            TR = TR @ infsimal
        else:
            D11, D12, D21, D22 = four_matrix(N, D_M_f, -0.5 * dx)
            infsimal[0:N, 0:N] = 0.5 * U @ D11 @ Udag_f
            infsimal[0:N, N:2*N] = 0.5 * U @ D12 @ Udag_f
            infsimal[N:2*N, 0:N] = 0.5 * U @ D21 @ Udag_f
            infsimal[N:2*N, N:2*N] = 0.5 * U @ D22 @ Udag_f
            TR = TR @ infsimal

        return TL, TR


def kernel_TM_np_exp_ceter(x_mesh, dx, N, direct, M_list):
    """
    Compute transfer matrix from center region (without boundary handling).
    Similar to kernel_TM_np_exp but starts from index 0 instead of 1.
    """
    infsimal = np.zeros((2 * N, 2 * N), dtype=np.complex128)
    
    if direct == 0 or direct == 1:
        T = np.eye(2 * N)
        for index in range(0, x_mesh):
            if direct == 0:
                i = index
            elif direct == 1:
                i = x_mesh - index
                dx = -dx
                
            D_M, U = eigh_nb(M_list[i, :, :])
            Udag = U.conjugate().T
            D11, D12, D21, D22 = four_matrix(N, D_M, dx)
            
            infsimal[0:N, 0:N] = 0.5 * U @ D11 @ Udag
            infsimal[0:N, N:2*N] = 0.5 * U @ D12 @ Udag
            infsimal[N:2*N, 0:N] = 0.5 * U @ D21 @ Udag
            infsimal[N:2*N, N:2*N] = 0.5 * U @ D22 @ Udag
            T = infsimal @ T
        return T

    elif direct == 2:
        TL = np.eye(2 * N)
        TR = np.eye(2 * N)
        
        for i in range(0, x_mesh):
            D_M, U = eigh_nb(M_list[i, :, :])
            Udag = U.conjugate().T
            D11, D12, D21, D22 = four_matrix(N, D_M, dx)
            
            infsimal[0:N, 0:N] = 0.5 * U @ D11 @ Udag
            infsimal[0:N, N:2*N] = 0.5 * U @ D12 @ Udag
            infsimal[N:2*N, 0:N] = 0.5 * U @ D21 @ Udag
            infsimal[N:2*N, N:2*N] = 0.5 * U @ D22 @ Udag
            TL = infsimal @ TL
            
            D11, D12, D21, D22 = four_matrix(N, D_M, -dx)
            infsimal[0:N, 0:N] = 0.5 * U @ D11 @ Udag
            infsimal[0:N, N:2*N] = 0.5 * U @ D12 @ Udag
            infsimal[N:2*N, 0:N] = 0.5 * U @ D21 @ Udag
            infsimal[N:2*N, N:2*N] = 0.5 * U @ D22 @ Udag
            TR = TR @ infsimal
       
        return TL, TR


def transfer_matrix(parameter_M, lib='np', side='both', Efield=False):
    """
    Compute transfer matrix.
    
    Parameters:
    -----------
    parameter_M : list
        [x_mesh, dx, N, M_list] where:
        - x_mesh: number of spatial points
        - dx: spatial step
        - N: total frequency cutoff
        - M_list: 3D array of M matrices
    lib : str
        Library to use ('np' for numpy)
    side : str
        Calculation direction: 'left', 'right', or 'both'
        'both' mode merges two calculations for maximum precision (recommended)
    Efield : bool
        Whether electric field is present
        
    Returns:
    --------
    Transfer matrix or tuple of transfer matrices
    """
    x_mesh, dx, N, M_list = parameter_M

    if lib == 'np':
        if side == 'left':
            return kernel_TM_np_exp(x_mesh, dx, N, 0, M_list)
        elif side == 'right':
            return kernel_TM_np_exp(x_mesh, dx, N, 1, M_list)
        elif side == 'both':
            return kernel_TM_np_exp(x_mesh, dx, N, 2, M_list, Efiled=Efield)


def diagonalize_TL_TR(N, TL, TR, threshold=1e-10, plot=False):
    """
    Diagonalize transfer matrices computed from both sides.
    This function combines left and right transfer matrices to extract accurate eigenvalues and eigenvectors.
    
    Parameters:
    -----------
    N : int
        Half dimension of the system
    TL : ndarray
        Transfer matrix from left
    TR : ndarray
        Transfer matrix from right
    threshold : float
        Threshold for determining unit modulus eigenvalues
    plot : bool
        Whether to plot eigenvalue distribution
        
    Returns:
    --------
    eigvals_TL_sorted : ndarray
        Sorted eigenvalues
    eigvecs_TL_sorted : ndarray
        Sorted eigenvectors
    count : int
        Number of edge states (eigenvalues with |λ| ≠ 1)
    """
    # Compute eigenvalues and eigenvectors of TR
    eigvals_TR, eigvecs_TR = eig_nb(TR)
    
    # Create J matrix (symplectic structure)
    J = np.block([[np.zeros((N, N)), np.eye(N)], [-np.eye(N), np.zeros((N, N))]])
    
    # Compute eigenvalues and eigenvectors of TL
    eigvals_TL, eigvecs_TL = eig_nb(TL)

    r_L = np.abs(eigvals_TL)
    theta_L = np.angle(eigvals_TL)
    lg_r_L = np.log(r_L)

    if plot == True:
        # Plot scatter diagram of eigenvalues in (log|r|, θ) space
        plt.scatter(lg_r_L, theta_L, c='b', s=20, alpha=0.3)
        plt.xlabel('lg(r)')
        plt.ylabel('theta')
        plt.title('eigvalues')
        plt.grid(True)

    r_R = np.abs(eigvals_TR)
    theta_R = -np.angle(eigvals_TR)
    lg_r_R = -np.log(r_R)

    if plot == True:
        plt.scatter(lg_r_R, theta_R, c='r', s=20, alpha=0.3)
        plt.xlabel('lg(r)')
        plt.ylabel('theta')
        plt.title('eigvalues')
        plt.grid(True)

    if plot == True:
        plt.show()
        plt.scatter(lg_r_L, theta_L, c='b', s=20, alpha=0.3)
        plt.scatter(lg_r_R, theta_R, c='r', s=20, alpha=0.3)
        plt.xlabel('lg(r)')
        plt.ylabel('theta')
        plt.title('eigvalues')
        plt.grid(True)
        plt.xlim(-0.01, 0.01)
        plt.show()
        
        plt.scatter(lg_r_L, theta_L, c='b', s=20, alpha=0.3)
        plt.scatter(lg_r_R, theta_R, c='r', s=20, alpha=0.3)
        plt.xlabel('lg(r)')
        plt.ylabel('theta')
        plt.title('eigvalues')
        plt.grid(True)
        plt.xlim(-0.001, 0.001)
        plt.show()
        
        plt.scatter(lg_r_L, theta_L, c='b', s=20, alpha=0.3)
        plt.scatter(lg_r_R, theta_R, c='r', s=20, alpha=0.3)
        plt.xlabel('lg(r)')
        plt.ylabel('theta')
        plt.title('eigvalues')
        plt.grid(True)
        plt.xlim(-0.0001, 0.0001)
        plt.show()
    
    # Sort eigenvalues by modulus
    sorted_indices_TR = np.argsort(np.abs(eigvals_TR))
    eigvals_TR_sorted = eigvals_TR[sorted_indices_TR]
    eigvecs_TR_sorted = eigvecs_TR[:, sorted_indices_TR]
    
    sorted_indices_TL = np.argsort(np.abs(eigvals_TL))
    eigvals_TL_sorted = eigvals_TL[sorted_indices_TL]
    eigvecs_TL_sorted = eigvecs_TL[:, sorted_indices_TL]
    
    # Determine the number of eigenvalues with unit modulus
    count = 0
    for i in range(2*N):
        if np.abs(eigvals_TL_sorted[i]) > 1 - threshold:
            break
        count += 1

    # Replace eigenvalues and eigenvectors for non-unit modulus states
    # Use TR eigenvectors for decaying modes (improves numerical stability)
    for i in range(count):
        eigvals_TL_sorted[i] = 1 / eigvals_TR_sorted[-i-1]
        eigvecs_TL_sorted[:, i] = eigvecs_TR_sorted[:, -i-1]
    
    # Normalize unit modulus eigenvectors and sort by current
    current = np.zeros(2*N - 2*count)
    for i in range(count, 2*N - count):
        norm = np.abs(np.dot(eigvecs_TL_sorted[:, i].conjugate(), J @ eigvecs_TL_sorted[:, i]))
        current[i-count] = np.real(-0.5j * np.dot(eigvecs_TL_sorted[:, i].conjugate(), J @ eigvecs_TL_sorted[:, i]))
        eigvecs_TL_sorted[:, i] = eigvecs_TL_sorted[:, i] / np.sqrt(norm)

    # Sort Bloch states by current (for proper ordering)
    sorted_indices_Bloch = np.argsort(-current) + count
    eigvals_TL_sorted_Bloch = eigvals_TL_sorted[sorted_indices_Bloch]
    eigvecs_TL_sorted_Bloch = eigvecs_TL_sorted[:, sorted_indices_Bloch]
    
    eigvals_TL_sorted[count:2*N-count] = eigvals_TL_sorted_Bloch[:]
    eigvecs_TL_sorted[:, count:2*N-count] = eigvecs_TL_sorted_Bloch[:]

    # Reverse ordering for proper state arrangement
    eigvecs_TL_sorted[:, count:N] = eigvecs_TL_sorted[:, count:N][:, ::-1]
    eigvals_TL_sorted[count:N] = eigvals_TL_sorted[count:N][::-1]
    eigvecs_TL_sorted[:, N:2*N-count] = eigvecs_TL_sorted[:, N:2*N-count][:, ::-1]
    eigvals_TL_sorted[N:2*N-count] = eigvals_TL_sorted[N:2*N-count][::-1]
    eigvecs_TL_sorted[:, N:2*N] = eigvecs_TL_sorted[:, N:2*N][:, ::-1]
    eigvals_TL_sorted[N:2*N] = eigvals_TL_sorted[N:2*N][::-1]

    r = np.abs(eigvals_TL_sorted)
    theta = np.angle(eigvals_TL_sorted)
    lg_r = np.log(r)

    # Plot final eigenvalue distribution
    if plot == True:
        plt.scatter(lg_r, theta, c='g', s=20, alpha=0.5)
        plt.xlabel('lg(r)')
        plt.ylabel('theta')
        plt.title('eigvalues')
        plt.grid(True)
        plt.show()
    
    return eigvals_TL_sorted, eigvecs_TL_sorted, count


def diagonalize_T(N, T, threshold=1e-10):
    """
    Diagonalize single transfer matrix (left mode).
    Simpler version for when only one direction is available.
    
    Parameters:
    -----------
    N : int
        Half dimension
    T : ndarray
        Transfer matrix
    threshold : float
        Threshold for unit modulus determination
        
    Returns:
    --------
    eigvals_T_sorted, eigvecs_T_sorted, count
    """
    # Create J matrix
    J = np.block([[np.zeros((N, N)), np.eye(N)], [-np.eye(N), np.zeros((N, N))]])
    
    # Compute eigenvalues and eigenvectors
    eigvals_T, eigvecs_T = eig_nb(T)

    r = np.abs(eigvals_T)
    theta = np.angle(eigvals_T)
    lg_r = np.log(r)

    sorted_indices_T = np.argsort(np.abs(eigvals_T))
    eigvals_T_sorted = eigvals_T[sorted_indices_T]
    eigvecs_T_sorted = eigvecs_T[:, sorted_indices_T]

    # Determine number of unit modulus eigenvalues
    count = 0
    for i in range(2*N):
        if np.abs(eigvals_T_sorted[i]) > 1 - threshold:
            break
        count += 1

    # Compute current and sort Bloch states
    current = np.zeros(2*N - 2*count)
    for i in range(count, 2*N - count):
        current[i-count] = np.real(-0.5j * np.dot(eigvecs_T_sorted[:, i].conjugate().T, J @ eigvecs_T_sorted[:, i]))
        
    sorted_indices_Bloch = np.argsort(-current)
    eigvals_T_sorted_Bloch = eigvals_T_sorted[sorted_indices_Bloch]
    eigvecs_T_sorted_Bloch = eigvecs_T_sorted[:, sorted_indices_Bloch]
    
    eigvals_T_sorted[count:2*N-count] = eigvals_T_sorted_Bloch
    eigvecs_T_sorted[:, count:2*N-count] = eigvecs_T_sorted_Bloch

    return eigvals_T_sorted, eigvecs_T_sorted, count


def compute_left_eigenvectors(Y, N_edge_oneside, eigenvalues=None):
    """
    Compute left eigenvectors with proper handling of:
    - Non-unit modulus modes: Y[:, 0:N_edge_oneside] and Y[:, N:N+N_edge_oneside] appear in pairs
    - Self-dual unit modulus modes: Y[:, N_edge_oneside:N] and Y[:, N+N_edge_oneside:2*N]

    Parameters:
    -----------
    Y : ndarray
        2N x 2N right eigenvector matrix (column vectors)
    N_edge_oneside : int
        Number of edge states with |λ|≠1 on one side
    eigenvalues : ndarray, optional
        Corresponding eigenvalues

    Returns:
    --------
    Y_sharp : ndarray
        Left eigenvector matrix (row vectors)
    """
    # Enforce data type
    Y = np.array(Y, dtype=np.complex128)
    if eigenvalues is not None:
        eigenvalues = np.array(eigenvalues, dtype=np.complex128)
    
    N = Y.shape[0] // 2
    J = np.block([[np.zeros((N, N), dtype=np.complex128), np.eye(N, dtype=np.complex128)],
                  [-np.eye(N, dtype=np.complex128), np.zeros((N, N), dtype=np.complex128)]])
    
    Y_sharp = np.zeros((2*N, 2*N), dtype=np.complex128)

    # (1) Handle non-unit modulus modes (appear in pairs)
    for i in range(N_edge_oneside):
        # Pair eigenvectors: Y[:, i] and Y[:, N+i]
        Y_i = Y[:, i]
        Y_pair = Y[:, N + i]
        
        if eigenvalues is not None:
            # Verify pairing relation |λ_i * λ_pair| ≈ 1
            assert np.isclose(np.abs(eigenvalues[i] * eigenvalues[N + i]), 1, atol=1e-8), \
                f"Pairing failed: λ[{i}]={eigenvalues[i]}, λ[{N+i}]={eigenvalues[N+i]}"
        
        # Compute left eigenvectors
        norm_ij = Y_pair.conj().T @ J @ Y_i
        norm_ji = Y_i.conj().T @ J @ Y_pair
        
        Y_sharp[i] = (Y_pair.conj().T @ J) / norm_ij
        Y_sharp[N + i] = (Y_i.conj().T @ J) / norm_ji

    # (2) Handle self-dual unit modulus modes (two segments)
    # First segment: Y[:, N_edge_oneside:N]
    for k in range(N_edge_oneside, N):
        Y_k = Y[:, k]
        norm_k = Y_k.conj().T @ J @ Y_k
        if np.abs(norm_k) < 1e-8:
            raise ValueError(f"Unit modulus mode {k} has near-zero normalization factor")
        Y_sharp[k] = (Y_k.conj().T @ J) / norm_k
    
    # Second segment: Y[:, N+N_edge_oneside:2*N]
    for m in range(N + N_edge_oneside, 2*N):
        Y_m = Y[:, m]
        norm_m = Y_m.conj().T @ J @ Y_m
        if np.abs(norm_m) < 1e-8:
            raise ValueError(f"Unit modulus mode {m} has near-zero normalization factor")
        Y_sharp[m] = (Y_m.conj().T @ J) / norm_m

    # Verify orthogonality
    error = np.linalg.norm(Y @ Y_sharp - np.eye(2*N, dtype=np.complex128))
    print(f"Orthogonality error: {error:.2e}")

    return Y_sharp


def categorize_states(eigvals, eigvecs, NLedge, N):
    """
    Categorize eigenstates into Bloch states and edge states.
    
    Parameters:
    -----------
    eigvals : ndarray
        Eigenvalues
    eigvecs : ndarray
        Eigenvectors
    NLedge : int
        Number of left edge states
    N : int
        Half dimension
        
    Returns:
    --------
    dict : Dictionary containing:
        - "Bloch States" and "Bloch Eigenvalues"
        - "Left Edge States" and "Left Edge Eigenvalues"
        - "Right Edge States" and "Right Edge Eigenvalues"
    """
    bloch_states = []
    ledge_states = []
    redge_states = []
    bloch_eigenvalues = []
    ledge_eigenvalues = []
    redge_eigenvalues = []    

    for i in range(NLedge):
        ledge_states.append(eigvecs[:, i])
        ledge_eigenvalues.append(eigvals[i])

    for i in range(NLedge, N):
        bloch_states.append(eigvecs[:, i])
        bloch_eigenvalues.append(eigvals[i])

    for i in range(N, N+NLedge):
        redge_states.append(eigvecs[:, i])
        redge_eigenvalues.append(eigvals[i])

    for i in range(N+NLedge, 2*N):
        bloch_states.append(eigvecs[:, i])
        bloch_eigenvalues.append(eigvals[i])

    return {
        "Bloch States": bloch_states,
        "Bloch Eigenvalues": bloch_eigenvalues,
        "Left Edge States": ledge_states,
        "Left Edge Eigenvalues": ledge_eigenvalues,
        "Right Edge States": redge_states,
        "Right Edge Eigenvalues": redge_eigenvalues
    }


def solve_channels(bloch_states, bloch_eigenvalues, ledge_states, ledge_eigenvalues, 
                   redge_states, redge_eigenvalues, Ain, Bin, L, frq_cutoff_oneside, E, omega):
    """
    Directly solve for scattering amplitudes using Bloch and edge states.
    
    Parameters:
    -----------
    bloch_states, bloch_eigenvalues : Bloch mode information
    ledge_states, ledge_eigenvalues : Left edge state information
    redge_states, redge_eigenvalues : Right edge state information
    Ain, Bin : Input vectors
    L : System length
    frq_cutoff_oneside : Frequency cutoff (one side)
    E, omega : Energy and frequency parameters
    
    Returns:
    --------
    Aout, Bout : Output vectors
    C : Bloch mode amplitudes
    DL, DR : Left and right edge mode amplitudes
    """
    n_B = int(len(bloch_eigenvalues) / 2)
    n_E = len(ledge_eigenvalues)
    n = n_B + n_E

    # Construct the matrix and vector
    A = np.zeros((4 * n, 4 * n), dtype=np.complex128)
    b = np.zeros(4 * n, dtype=np.complex128)

    # Fill in the matrix and vector
    for i in range(n):
        b[i] = Ain[i]
        b[i+n] = Ain[i]
        b[i+2*n] = Bin[i]
        b[i+3*n] = Bin[i]
        
        for j in range(n):
            if j == i:
                A[i, j] = -1
                A[i+n, j] = 1
                A[i+2*n, j+n] = -1
                A[i+3*n, j+n] = 1

        for j in range(2*n_B):
            A[i, j+2*n] = bloch_states[j][i]
            A[i+n, j+2*n] = bloch_states[j][i+n] / (1j * np.sqrt(2 * (E + (i - frq_cutoff_oneside) * omega) + 0j))
            A[i+2*n, j+2*n] = bloch_states[j][i] * np.exp(1j * np.angle(bloch_eigenvalues[j]) * L)
            A[i+3*n, j+2*n] = bloch_states[j][i+n] * np.exp(1j * np.angle(bloch_eigenvalues[j]) * L) / \
                             (-1j * np.sqrt(2 * (E + (i - frq_cutoff_oneside) * omega) + 0j))
            
        for j in range(n_E):
            A[i, j+2*n+2*n_B] = ledge_states[j][i]
            A[i, j+2*n+2*n_B+n_E] = (1/redge_eigenvalues[j])**L * redge_states[j][i]
            
            A[i+n, j+2*n+2*n_B] = ledge_states[j][i+n] / (1j * np.sqrt(2 * (E + (i - frq_cutoff_oneside) * omega) + 0j))
            A[i+n, j+2*n+2*n_B+n_E] = (1/redge_eigenvalues[j])**L * redge_states[j][i+n] / \
                                      (1j * np.sqrt(2 * (E + (i - frq_cutoff_oneside) * omega) + 0j))
            
            A[i+2*n, j+2*n+2*n_B] = ledge_states[j][i] * ledge_eigenvalues[j]**L
            A[i+2*n, j+2*n+2*n_B+n_E] = redge_states[j][i]
            
            A[i+3*n, j+2*n+2*n_B] = ledge_states[j][i+n] * ledge_eigenvalues[j]**L / \
                                   (-1j * np.sqrt(2 * (E + (i - frq_cutoff_oneside) * omega) + 0j))
            A[i+3*n, j+2*n+2*n_B+n_E] = redge_states[j][i+n] / \
                                        (-1j * np.sqrt(2 * (E + (i - frq_cutoff_oneside) * omega) + 0j))

    # Solve the linear system
    x = np.linalg.solve(A, b)

    # Extract solutions
    Aout = x[:n]
    Bout = x[n:2*n]
    C = x[2*n:2*n+2*n_B]
    DL = x[2*n+2*n_B:2*n+2*n_B+n_E]
    DR = x[2*n+2*n_B+n_E:]

    return Aout, Bout, C, DL, DR


def gen_free_space(N_1side, E, omega, mu=0.5):
    """
    Generate free space basis states (incoming and outgoing waves).
    
    Parameters:
    -----------
    N_1side : int
        One-sided frequency cutoff
    E : float
        Energy
    omega : float
        Frequency
    mu : float
        Mass (default 0.5)
        
    Returns:
    --------
    Fp, Fp_sharp, Fn, Fn_sharp : Basis states and their duals
    """
    N = N_1side * 2 + 1
    Fp = np.zeros((2*N, N), dtype=np.complex128)
    Fn = np.zeros((2*N, N), dtype=np.complex128)
    Fp_sharp = np.zeros((N, 2*N), dtype=np.complex128)
    Fn_sharp = np.zeros((N, 2*N), dtype=np.complex128)
    
    for i in range(N):
        n = i - N_1side
        Ek = E + n * omega
        
        if Ek >= 0:
            k = 1j * np.sqrt(2 * mu * Ek)
            Fp[i, i] = 1 / np.sqrt(np.abs(k))
            Fp[i+N, i] = k / np.sqrt(np.abs(k))
            Fp_sharp[i, i] = k / np.sqrt(np.abs(k)) * (-0.5j)
            Fp_sharp[i, i+N] = 1 / np.sqrt(np.abs(k)) * (-0.5j)
            Fn[i, i] = 1 / np.sqrt(np.abs(k))
            Fn[i+N, i] = -k / np.sqrt(np.abs(k))
            Fn_sharp[i, i] = -k / np.sqrt(np.abs(k)) * (0.5j)
            Fn_sharp[i, i+N] = 1 / np.sqrt(np.abs(k)) * (0.5j)
        else:
            k = -np.sqrt(-2 * mu * Ek)
            Fp[i, i] = 1 / np.sqrt(np.abs(k))
            Fp[i+N, i] = k / np.sqrt(np.abs(k))
            Fp_sharp[i, i] = k / np.sqrt(np.abs(k)) * (-0.5)
            Fp_sharp[i, i+N] = 1 / np.sqrt(np.abs(k)) * (-0.5)
            Fn[i, i] = 1 / np.sqrt(np.abs(k))
            Fn[i+N, i] = -k / np.sqrt(np.abs(k))
            Fn_sharp[i, i] = -k / np.sqrt(np.abs(k)) * (0.5)
            Fn_sharp[i, i+N] = 1 / np.sqrt(np.abs(k)) * (0.5)

    print('free space orthogonal')
    print(np.linalg.norm(Fn @ Fn_sharp + Fp @ Fp_sharp - np.eye(2*N, dtype=np.complex128)))

    return (Fp, Fp_sharp, Fn, Fn_sharp)


def gen_space(eigvecs_sorted, N_edge_oneside):
    """
    Generate space basis from sorted eigenvectors.
    
    Parameters:
    -----------
    eigvecs_sorted : ndarray
        Sorted eigenvectors
    N_edge_oneside : int
        Number of edge states per side
        
    Returns:
    --------
    Tuple of basis states: (plus states, plus duals, minus states, minus duals)
    """
    N = np.size(eigvecs_sorted, 0) // 2
    sharp = compute_left_eigenvectors(eigvecs_sorted, N_edge_oneside)
    sharp_plus = sharp[0:N, :]
    sharp_minus = sharp[N:2*N, :]

    return (eigvecs_sorted[:, 0:N], sharp_plus, eigvecs_sorted[:, N:2*N], sharp_minus)


def semi_inf(A_space, B_space, side='left'):
    """
    Compute transmission and reflection for semi-infinite system.
    
    Parameters:
    -----------
    A_space, B_space : tuple
        Space basis tuples (plus, plus_sharp, minus, minus_sharp)
    side : str
        'left' or 'right'
        
    Returns:
    --------
    T, R : Transmission and reflection matrices
    """
    A_plus, A_plus_sharp, A_minus, A_minus_sharp = A_space
    B_plus, B_plus_sharp, B_minus, B_minus_sharp = B_space
    
    if side == 'left':
        TL = np.linalg.inv(np.dot(A_plus_sharp, B_plus))
        RL = np.dot(np.dot(A_minus_sharp, B_plus), TL)
        return TL, RL
    elif side == 'right':
        TR = np.linalg.inv(np.dot(B_minus_sharp, A_minus))
        RR = np.dot(np.dot(B_plus_sharp, A_minus), TR)
        return TR, RR


def tri_region_exact(L_space, M_space, D, L, R_space, side='left'):
    """
    Exact solution for three-region system (semi-infinite | finite | semi-infinite).
    
    Parameters:
    -----------
    L_space, M_space, R_space : Space bases for left, middle, right regions
    D : tuple
        (D_plus, D_minus) eigenvalues for middle region
    L : int
        Length of middle region
    side : str
        Computation direction
        
    Returns:
    --------
    T_total, R_total : Total transmission and reflection
    """
    N = np.size(D[0], 0)
    D_plus, D_minus = D
    D_plus_powered = np.diag(D_plus**L)
    D_minus_powered = np.diag(D_minus**(-L))
    
    if side == 'left':
        T1, R1 = semi_inf(L_space, M_space, side='left')
        T2, R2 = semi_inf(L_space, M_space, side='right')
        T3, R3 = semi_inf(M_space, R_space, side='left')

        # Compute T_total
        inv_term_T = np.linalg.inv(np.eye(N) - D_plus_powered @ R2 @ D_minus_powered @ R3)
        T_total = T3 @ inv_term_T @ D_plus_powered @ T1

        # Compute R_total
        inv_term_R = np.linalg.inv(np.eye(N) - D_minus_powered @ R3 @ D_plus_powered @ R2)
        R_total = R1 + T2 @ inv_term_R @ D_minus_powered @ R3 @ D_plus_powered @ T1
    
    elif side == 'right':
        T1, R1 = semi_inf(M_space, R_space, side='right')
        T2, R2 = semi_inf(M_space, R_space, side='left')
        T3, R3 = semi_inf(L_space, M_space, side='right')

        # Compute T_total
        inv_term_T = np.linalg.inv(np.eye(N) - D_minus_powered @ R2 @ D_plus_powered @ R3)
        T_total = T3 @ inv_term_T @ D_minus_powered @ T1

        # Compute R_total
        inv_term_R = np.linalg.inv(np.eye(N) - D_plus_powered @ R3 @ D_minus_powered @ R2)
        R_total = R1 + T2 @ inv_term_R @ D_plus_powered @ R3 @ D_minus_powered @ T1

    return T_total, R_total


def S_matrix_merge(T1L, T1R, R1L, R1R, T2L, T2R, R2L, R2R):
    """
    Merge two S-matrices in series.
    
    Parameters:
    -----------
    T1L, T1R, R1L, R1R : First S-matrix components
    T2L, T2R, R2L, R2R : Second S-matrix components
    
    Returns:
    --------
    TL, TR, RL, RR : Merged S-matrix components
    """
    N = np.size(T1L, 0)

    L_inv_term = np.linalg.inv(np.eye(N) - np.dot(R1R, R2L))
    TL = T2L @ L_inv_term @ T1L
    RL = R1L + T1R @ R2L @ L_inv_term @ T1L

    R_inv_term = np.linalg.inv(np.eye(N) - np.dot(R2L, R1R))
    TR = T1R @ R_inv_term @ T2R
    RR = R2R + T2L @ R1R @ R_inv_term @ T2R

    return TL, TR, RL, RR


def Smatrix_merge2(T1L, T1R, R1L, R1R, T2L, T2R, R2L, R2R, D, L):
    """
    Merge two S-matrices with intermediate propagation phase.
    First S-matrix connects AB spaces, second connects BC spaces.
    Requires consistent intermediate space B.
    
    Parameters:
    -----------
    T1L, T1R, R1L, R1R : First S-matrix
    T2L, T2R, R2L, R2R : Second S-matrix
    D : tuple (D_plus, D_minus)
        Eigenvalues for intermediate region
    L : float
        Length of intermediate region
        
    Returns:
    --------
    T_total_L, T_total_R, R_total_L, R_total_R : Merged S-matrix
    """
    N = np.size(D[0], 0)
    D_plus, D_minus = D
    D_plus_powered = np.diag(D_plus**L)
    D_minus_powered = np.diag(D_minus**(-L))

    inv_term_T = np.linalg.inv(np.eye(N) - D_plus_powered @ R1R @ D_minus_powered @ R2L)
    T_total_L = T2L @ inv_term_T @ D_plus_powered @ T1L

    inv_term_R = np.linalg.inv(np.eye(N) - D_minus_powered @ R2L @ D_plus_powered @ R1R)
    R_total_L = R1L + T1R @ inv_term_R @ D_minus_powered @ R2R @ D_plus_powered @ T1L

    inv_term_T = np.linalg.inv(np.eye(N) - D_minus_powered @ R2L @ D_plus_powered @ R1R)
    T_total_R = T1R @ inv_term_T @ D_minus_powered @ T2R

    inv_term_R = np.linalg.inv(np.eye(N) - D_plus_powered @ R1R @ D_minus_powered @ R2L)
    R_total_R = R2R + T2L @ inv_term_R @ D_plus_powered @ R1L @ D_minus_powered @ T2R

    return T_total_L, T_total_R, R_total_L, R_total_R


def Smatrix_series(mu, V1, V2, shift, nd1, nd2, d, omega, x_mesh, frqcutoff_oneside, E, N_conduct):
    """
    Generate series of S-matrices for conductance calculations.
    Creates multiple S-matrices with gradually increasing potential strength.
    
    Parameters:
    -----------
    mu : float
        Mass
    V1, V2 : float
        Potential amplitudes
    shift : float
        Potential phase shift
    nd1, nd2 : int
        Mode numbers
    d : float
        Period
    omega : float
        Frequency
    x_mesh : int
        Spatial mesh points
    frqcutoff_oneside : int
        One-sided frequency cutoff
    E : float
        Energy
    N_conduct : int
        Number of conductance steps
        
    Returns:
    --------
    left_side_package, right_side_package, D_scattering :
        Packages containing S-matrices for left and right sides, plus scattering eigenvalues
    """
    N = 2 * frqcutoff_oneside + 1
    TL_left_series = []
    TR_left_series = []
    RL_left_series = []
    RR_left_series = []
    D_left_series = []

    TL_right_series = []
    TR_right_series = []
    RL_right_series = []
    RR_right_series = []
    Fspace = gen_free_space(frqcutoff_oneside, E, omega)

    for index in range(N_conduct + 1):
        if index == 0:
            print('index==0')
            Lspace = Fspace
            factor = (index + 1) / (N_conduct + 1)
            parameter_M = potentials.double_wave(mu, factor*V1, factor*V2, factor*shift, nd1, nd2, d, omega, 
                                    x_mesh, frqcutoff_oneside, E)
            TrsfL, TrsfR = transfer_matrix(parameter_M, side='both')
            eigvals, eigvecs, NLedge = diagonalize_TL_TR(N, TrsfL, TrsfR)
            print('********************')
            print(np.log(np.abs(eigvals)))
            print('********************')
            
            Rspace = gen_space(eigvecs, NLedge)
            D = (eigvals[0:N], eigvals[N:2*N])
            D_left_series.append(D)
            
            TL, RL = semi_inf(Lspace, Rspace, side='left')
            TL_left_series.append(TL)
            RL_left_series.append(RL)
            TR, RR = semi_inf(Lspace, Rspace, side='right')
            TR_left_series.append(TR)
            RR_left_series.append(RR)

            TL, RL = semi_inf(Rspace, Lspace, side='left')
            TL_right_series.append(TL)
            RL_right_series.append(RL)
            TR, RR = semi_inf(Rspace, Lspace, side='right')
            TR_right_series.append(TR)
            RR_right_series.append(RR)

            Lspace = Rspace
        else:
            factor = (index + 1) / (N_conduct + 1)
            parameter_M = potentials.double_wave(mu, factor*V1, factor*V2, factor*shift, nd1, nd2, d, omega, 
                                    x_mesh, frqcutoff_oneside, E)
            TrsfL, TrsfR = transfer_matrix(parameter_M, side='both')
            eigvals, eigvecs, NLedge = diagonalize_TL_TR(N, TrsfL, TrsfR)
            print('####################')
            print(np.log(np.abs(eigvals)))
            print('####################')
            
            Rspace = gen_space(eigvecs, NLedge)
            D = (eigvals[0:N], eigvals[N:2*N])
            if index != N_conduct:
                D_left_series.append(D)
            else:
                D_scattering = D

            TL, RL = semi_inf(Lspace, Rspace, side='left')
            TL_left_series.append(TL)
            RL_left_series.append(RL)
            TR, RR = semi_inf(Lspace, Rspace, side='right')
            TR_left_series.append(TR)
            RR_left_series.append(RR)

            TL, RL = semi_inf(Rspace, Lspace, side='left')
            TL_right_series.append(TL)
            RL_right_series.append(RL)
            TR, RR = semi_inf(Rspace, Lspace, side='right')
            TR_right_series.append(TR)
            RR_right_series.append(RR)

            Lspace = Rspace

    D_right_series = D_left_series[::-1]
    TL_right_series.reverse()
    TR_right_series.reverse()
    RL_right_series.reverse()
    RR_right_series.reverse()

    left_side_package = (TL_left_series, TR_left_series, RL_left_series, RR_left_series, D_left_series)
    right_side_package = (TL_right_series, TR_right_series, RL_right_series, RR_right_series, D_right_series)

    return left_side_package, right_side_package, D_scattering


def Smatrix_from_series(Spackage):
    """
    Construct complete S-matrix from series of S-matrices.
    
    Parameters:
    -----------
    Spackage : tuple
        (TL_series, TR_series, RL_series, RR_series, D_series)
        
    Returns:
    --------
    TL, TR, RL, RR : Final S-matrix components
    """
    TL_series, TR_series, RL_series, RR_series, D_series = Spackage
    N_conduct = len(TL_series) - 1
    TL = TL_series[0]
    RL = RL_series[0]
    TR = TR_series[0]
    RR = RR_series[0]

    for i in range(N_conduct):
        TL, TR, RL, RR = Smatrix_merge2(TL, TR, RL, RR, TL_series[i+1], TR_series[i+1], 
                                        RL_series[i+1], RR_series[i+1], D_series[i], 1)

    return TL, TR, RL, RR


def normalize_complex_array(arr, threshold=1e-10):
    """
    Normalize complex array: set modulus to 1 for elements with |arr| ≈ 1, keeping phase unchanged.
    
    Parameters:
    -----------
    arr : ndarray
        Input complex array
    threshold : float
        Threshold for determining if modulus is close to 1
        
    Returns:
    --------
    normalized_arr : ndarray
        Normalized complex array
    """
    # Compute modulus of each element
    magnitudes = np.abs(arr)
    
    # Compute phase of each element
    angles = np.angle(arr)
    
    # Determine which elements have modulus close to 1
    close_to_one = np.abs(magnitudes - 1) < threshold
    
    # For elements close to 1, set modulus to 1, keep phase; otherwise keep original
    normalized_magnitudes = np.where(close_to_one, 1, magnitudes)
    
    # Reconstruct complex array
    normalized_arr = normalized_magnitudes * np.exp(1j * angles)
    
    return normalized_arr


def seminf_kernel_nojulia(A, B, C, N):
    """
    Semi-infinite kernel without Julia (pure NumPy implementation).
    Solves: A + B @ R = C @ T
    
    Parameters:
    -----------
    A, B, C : ndarray
        Input matrices (2N × N)
    N : int
        Half dimension
        
    Returns:
    --------
    R, T : Reflection and transmission matrices
    """
    A1 = A[0:N, :]
    A2 = A[N:2*N, :]
    B1 = B[0:N, :]
    B2 = B[N:2*N, :]
    C1 = C[0:N, :]
    C2 = C[N:2*N, :]
    
    print('start inv')
    B1i = np.linalg.inv(B1)
    B2i = np.linalg.inv(B2)
    C1i = np.linalg.inv(C1)
    C2i = np.linalg.inv(C2)
    
    T = np.linalg.inv(B1i @ C1 - B2i @ C2) @ (B1i @ A1 - B2i @ A2)
    R = -np.linalg.inv(C1i @ B1 - C2i @ B2) @ (C1i @ A1 - C2i @ A2)
    print('finished inv')

    return R, T


def seminf_kernel(A, B, C, N):
    """
    Semi-infinite kernel using Julia for high-precision matrix inversion.
    Automatically switches to arbitrary precision when condition number is high.
    
    Parameters:
    -----------
    A, B, C : ndarray
        Input matrices (2N × N)
    N : int
        Half dimension
        
    Returns:
    --------
    R, T : Reflection and transmission matrices
    """
    # Split matrices into blocks
    A1 = A[0:N, :]
    A2 = A[N:2*N, :]
    B1 = B[0:N, :]
    B2 = B[N:2*N, :]
    C1 = C[0:N, :]
    C2 = C[N:2*N, :]

    print('Start computation in Julia')

    # Dynamically initialize Julia environment
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    
    # Define compute_R_T function (only executed on first call)
    if not hasattr(seminf_kernel, "_julia_initialized"):
        jl.eval("""
        using LinearAlgebra

        function compute_R_T(A1, A2, B1, B2, C1, C2)
            # Dynamic precision inversion function
            function safe_inv(matrix)
                cond_num = cond(matrix)
                println("Condition number: ", cond_num)
                
                extra_bits = 10  # Extra safety bits
                precision_bits = max(64, Int(ceil(log2(cond_num))) + extra_bits)
                
                println("Using precision: ", precision_bits, " bits")
                
                if precision_bits > 64
                    setprecision(precision_bits) do
                        big_matrix = Complex{BigFloat}.(matrix)  # Convert to arbitrary precision
                        ComplexF64.(inv(big_matrix))             # Convert back to double precision
                    end
                else
                    inv(matrix)  # Use double precision directly
                end
            end

            # Compute inverse matrices
            B1i = safe_inv(B1)
            B2i = safe_inv(B2)
            C1i = safe_inv(C1)
            C2i = safe_inv(C2)

            # Compute T and R
            T = inv(B1i * C1 - B2i * C2) * (B1i * A1 - B2i * A2)
            R = -inv(C1i * B1 - C2i * B2) * (C1i * A1 - C2i * A2)

            return R, T
        end
        """)
        seminf_kernel._julia_initialized = True

    # Call Julia function for computation
    R, T = jl.eval("compute_R_T")(A1, A2, B1, B2, C1, C2)

    print('Finished computation')

    return R, T


@jit(nopython=True)
def seminf_kernel_onebyone(A, B, C, N):
    """
    Optimized seminf_kernel implementation avoiding explicit matrix inversion.
    Solves the system column by column using Gaussian elimination.
    
    Parameters:
    -----------
    A, B, C : ndarray
        Input matrices (2N × N)
    N : int
        Half dimension
        
    Returns:
    --------
    R, T : Reflection and transmission matrices (N × N)
    """
    print('start semikernel')
    R = np.zeros((N, N), dtype=np.complex128)
    T = np.zeros((N, N), dtype=np.complex128)
    
    # Construct augmented matrix [B | -C], shape (2N, 2N)
    BC = np.zeros((2*N, 2*N), dtype=np.complex128)
    BC[:, :N] = B
    BC[:, N:] = -C
    
    for i in range(N):
        # Construct right-hand side vector: -A[:, i]
        b = -A[:, i]
        
        # Solve [B | -C] [R[:,i]; T[:,i]] = -A[:,i] via Gaussian elimination
        x = np.linalg.solve(BC, b)
        
        # Extract solution components
        R[:, i] = x[:N]
        T[:, i] = x[N:]
    print('finished semikernel')
    
    return R, T


def connect_transmatrix(parameter_connect):
    """
    Connect transfer matrices (wrapper function).
    
    Parameters:
    -----------
    parameter_connect : list
        [x_mesh, dx, N, M_list]
        
    Returns:
    --------
    Transfer matrix
    """
    x_mesh, dx, N, M_list = parameter_connect
    return kernel_TM_np_exp(x_mesh, dx, N, 0, M_list)


def seminf(mu, E, omega, N_1side, eigvecs_sorted, connect=None):
    """
    Compute semi-infinite transmission and reflection matrices.
    
    Parameters:
    -----------
    mu : float
        Mass
    E : float
        Energy
    omega : float
        Frequency
    N_1side : int
        One-sided frequency cutoff
    eigvecs_sorted : ndarray
        Sorted eigenvectors of the system
    connect : ndarray, optional
        Connection matrix to apply to free space states
        
    Returns:
    --------
    R1, R2, R3, R4, T1, T2, T3, T4 : Scattering matrices
    """
    N = N_1side * 2 + 1
    Fp = np.zeros((2*N, N), dtype=np.complex128)
    Fn = np.zeros((2*N, N), dtype=np.complex128)
    Sp = np.zeros((2*N, N), dtype=np.complex128)
    Sn = np.zeros((2*N, N), dtype=np.complex128)
    
    for i in range(N):
        n = i - N_1side
        Ek = E + n * omega
        
        if Ek >= 0:
            k = 1j * np.sqrt(2 * mu * Ek)
        else:
            k = -np.sqrt(-2 * mu * Ek)
        
        Fp[i, i] = 1 / np.sqrt(np.abs(k))
        Fp[i+N, i] = k / np.sqrt(np.abs(k))
        Fn[i, i] = 1 / np.sqrt(np.abs(k))
        Fn[i+N, i] = -k / np.sqrt(np.abs(k))

        if connect is not None:
            Fn = connect @ Fn
            Fp = connect @ Fp

        Sp = eigvecs_sorted[:, 0:N]
        Sn = eigvecs_sorted[:, N:2*N]
    
    R1, T1 = seminf_kernel(Fp, Fn, Sp, N)
    R2, T2 = seminf_kernel(Sn, Sp, Fn, N)
    R3, T3 = seminf_kernel(Sp, Sn, Fp, N)
    R4, T4 = seminf_kernel(Fn, Fp, Sn, N)

    return R1, R2, R3, R4, T1, T2, T3, T4


def laser_analogy(R1, R2, R3, T1, T2, T3, N, eigvals_sorted, L, NLedge, ignoledge=False):
    """
    Laser cavity analogy for computing total transmission and reflection.
    Models the system as a Fabry-Perot resonator with multiple reflections.
    
    Parameters:
    -----------
    R1, R2, R3 : ndarray
        Reflection matrices
    T1, T2, T3 : ndarray
        Transmission matrices
    N : int
        Dimension
    eigvals_sorted : ndarray
        Sorted eigenvalues
    L : float
        System length
    NLedge : int
        Number of edge states
    ignoledge : bool
        Whether to ignore edge state contributions
        
    Returns:
    --------
    T_total, R_total : Total transmission and reflection matrices
    """
    Dp = eigvals_sorted[0:N]
    Dn = eigvals_sorted[N:2*N]

    D1 = np.zeros(N, dtype=np.complex128)
    D2 = np.zeros(N, dtype=np.complex128)
    I = np.eye(N)
    
    # For Bloch states, use phase only (unit modulus)
    D1[NLedge:N] = np.exp(1j * L * np.angle(Dp[NLedge:N]))
    D2[NLedge:N] = np.exp(-1j * L * np.angle(Dn[NLedge:N]))
    
    # For edge states, include exponential decay/growth
    if ignoledge == False:
        D1[0:NLedge] = np.power(Dp[0:NLedge], L)
        D2[0:NLedge] = np.power(1/Dn[0:NLedge], L)

    D1 = np.diag(D1)
    D2 = np.diag(D2)
    
    # Sum infinite series of reflections (geometric series)
    T_total = T3 @ np.linalg.inv(I - D1 @ R2 @ D2 @ R3) @ D1 @ T1
    R_total = R1 + T2 @ np.linalg.inv(I - D2 @ R3 @ D1 @ R2) @ D2 @ R3 @ D1 @ T1

    return T_total, R_total


def current(Y):
    """
    Compute current for a given state.
    
    Parameters:
    -----------
    Y : ndarray
        State vector
        
    Returns:
    --------
    Current value
    """
    N = np.size(Y) // 2
    J = np.zeros((2*N, 2*N), dtype=np.complex128)
    J[N:2*N, 0:N] = np.eye(N)
    J[0:N, N:2*N] = -np.eye(N)
    return np.imag(np.dot(Y.conjugate(), np.imag(J @ Y)))


def seminf_2(mu, E, omega, N_1side, eigvecs_sorted):
    """
    Alternative semi-infinite calculation with reordered states.
    (Specific ordering for particular use cases)
    """
    N = N_1side * 2 + 1
    Fp = np.zeros((2*N, N), dtype=np.complex128)
    Fn = np.zeros((2*N, N), dtype=np.complex128)
    Sp = np.zeros((2*N, N), dtype=np.complex128)
    Sn = np.zeros((2*N, N), dtype=np.complex128)
    
    for i in range(N):
        n = i - N_1side
        Ek = E + n * omega
        
        if Ek >= 0:
            k = 1j * np.sqrt(2 * mu * Ek)
        else:
            k = -np.sqrt(-2 * mu * Ek)
            
        Fp[i, i] = 1 / np.sqrt(np.abs(k))
        Fp[i+N, i] = k / np.sqrt(np.abs(k))
        Fn[i, i] = 1 / np.sqrt(np.abs(k))
        Fn[i+N, i] = -k / np.sqrt(np.abs(k))

        Sp = eigvecs_sorted[:, 0:N]
        Sp[:, N_1side:N] = eigvecs_sorted[:, N:2*N-N_1side]
        Sn = eigvecs_sorted[:, N:2*N]
        Sn[:, 0:N-N_1side] = eigvecs_sorted[N_1side, N]
    
    R1, T1 = seminf_kernel(Fp, Fn, Sp, N)
    R2, T2 = seminf_kernel(Sp, Sn, Fp, N)
    R3, T3 = seminf_kernel(Sn, Sp, Fn, N)

    return R1, R2, R3, T1, T2, T3


def static_kernel_TM_np_exp(x_mesh, dx, mu, potential, E):
    """
    Compute transfer matrix for static (time-independent) potential.
    
    Parameters:
    -----------
    x_mesh : int
        Number of spatial mesh points
    dx : float
        Spatial step size
    mu : float
        Mass
    potential : ndarray
        Potential energy array
    E : float
        Energy
        
    Returns:
    --------
    TL, TR : Transfer matrices from left and right
    """
    TL = np.eye(2)
    TR = np.eye(2)
    infsimal = np.zeros((2, 2), dtype=np.complex128)
    M_list = 2 * mu * (-E + potential)
    M_list = M_list.astype(np.complex128)
    
    for i in range(1, x_mesh):
        if np.real(M_list[i]) >= 0:
            Msq = np.sqrt(M_list[i])
        else:
            Msq = 1j * np.sqrt(-M_list[i])

        e1 = np.exp(Msq * dx)
        e2 = 1 / e1
        D11 = e1 + e2
        
        if abs(Msq) > 1e-16:
            D12 = (e1 - e2) / Msq
        else:
            print('too small eigenvalue, using Taylor expansion')
            D12 = 2 * dx + (Msq**2 * dx**3) / 3 + (Msq**4 * dx**5) / 60
            
        D21 = (e1 - e2) * Msq
        D22 = D11
        infsimal[0, 0] = 0.5 * D11
        infsimal[0, 1] = 0.5 * D12
        infsimal[1, 0] = 0.5 * D21
        infsimal[1, 1] = 0.5 * D22
        TL = infsimal @ TL

        e1 = np.exp(-Msq * dx)
        e2 = 1 / e1
        D11 = e1 + e2
        
        if abs(Msq) > 1e-16:
            D12 = (e1 - e2) / Msq
        else:
            print('too small eigenvalue, using Taylor expansion')
            D12 = -2 * dx + (-Msq**2 * dx**3) / 3 + (-Msq**4 * dx**5) / 60
            
        D21 = (e1 - e2) * Msq
        D22 = D11
        infsimal[0, 0] = 0.5 * D11
        infsimal[0, 1] = 0.5 * D12
        infsimal[1, 0] = 0.5 * D21
        infsimal[1, 1] = 0.5 * D22
        TR = TR @ infsimal
    
    dx = 0.5 * dx
    if M_list[0] >= 0:
        Msq = np.sqrt(M_list[0])
    else:
        Msq = 1j * np.sqrt(-M_list[0])
        
    e1 = np.exp(Msq * dx)
    e2 = 1 / e1
    D11 = e1 + e2
    
    if abs(Msq) > 1e-16:
        D12 = (e1 - e2) / Msq
    else:
        print('too small eigenvalue, using Taylor expansion')
        D12 = 2 * dx + (Msq**2 * dx**3) / 3 + (Msq**4 * dx**5) / 60
        
    D21 = (e1 - e2) * Msq
    D22 = D11
    infsimal[0, 0] = 0.5 * D11
    infsimal[0, 1] = 0.5 * D12
    infsimal[1, 0] = 0.5 * D21
    infsimal[1, 1] = 0.5 * D22
    TL = infsimal @ TL @ infsimal

    e1 = np.exp(-Msq * dx)
    e2 = 1 / e1
    D11 = e1 + e2
    
    if abs(Msq) > 1e-16:
        D12 = (e1 - e2) / Msq
    else:
        print('too small eigenvalue, using Taylor expansion')
        D12 = -2 * dx + (-Msq**2 * dx**3) / 3 + (-Msq**4 * dx**5) / 60
        
    D21 = (e1 - e2) * Msq
    D22 = D11
    infsimal[0, 0] = 0.5 * D11
    infsimal[0, 1] = 0.5 * D12
    infsimal[1, 0] = 0.5 * D21
    infsimal[1, 1] = 0.5 * D22
    TR = infsimal @ TR @ infsimal

    return TL, TR


def static_transfer_matrix(x_mesh, dx, mu, potential, E, lib='np'):
    """
    Wrapper for static transfer matrix calculation.
    
    Parameters:
    -----------
    x_mesh, dx, mu, potential, E : Parameters for static system
    lib : str
        Library to use (currently only 'np' supported)
        
    Returns:
    --------
    TL, TR : Transfer matrices
    """
    if lib == 'np':
        return static_kernel_TM_np_exp(x_mesh, dx, mu, potential, E)


@jit(nopython=True)
def average_trans(N_MC, R1, R2, R3, T1, T2, T3, N_edge, N):
    """
    Monte Carlo average of transmission and reflection over random phases.
    Used for phase-averaged transport calculations.
    
    Parameters:
    -----------
    N_MC : int
        Number of Monte Carlo samples
    R1, R2, R3, T1, T2, T3 : Scattering matrices
    N_edge : int
        Number of edge states
    N : int
        Total dimension
        
    Returns:
    --------
    T_total_2, R_total_2 : Phase-averaged |T|² and |R|²
    """
    print('in MC')
    T1 = T1[N_edge:N, :]
    T2 = T2[:, N_edge:N]
    T3 = T3[:, N_edge:N]
    R2 = R2[N_edge:N, N_edge:N]
    R3 = R3[N_edge:N, N_edge:N]
    NB = N - N_edge
    T_total_2 = np.zeros((N, N))
    R_total_2 = np.zeros((N, N))
    I = np.eye(NB)
    
    for time in range(N_MC):
        # Random phase for each Bloch mode
        phases = np.random.uniform(0, 2*np.pi, NB)
        D1 = np.diag(np.exp(1j * phases))
        phases = np.random.uniform(0, 2*np.pi, NB)
        D2 = np.diag(np.exp(1j * phases))

        T_total_2 = T_total_2 + (np.abs(T3 @ np.linalg.inv(I - D1 @ R2 @ D2 @ R3) @ D1 @ T1)**2) / N_MC
        R_total_2 = R_total_2 + (np.abs(R1 + T2 @ np.linalg.inv(I - D2 @ R3 @ D1 @ R2) @ D2 @ R3 @ D1 @ T1)**2) / N_MC
        
    print('out MC')

    return T_total_2, R_total_2


def approx_average_trans(R1, R2, R3, T1, T2, T3, N_edge, N):
    """
    Approximate phase-averaged transmission (assumes phases are independent).
    Faster but less accurate than Monte Carlo averaging.
    
    Parameters:
    -----------
    R1, R2, R3, T1, T2, T3 : Scattering matrices
    N_edge : int
        Number of edge states
    N : int
        Total dimension
        
    Returns:
    --------
    Rtotal2, Ttotal2 : Approximate |R|² and |T|²
    """
    T1 = T1[N_edge:N, :]
    T2 = T2[:, N_edge:N]
    T3 = T3[:, N_edge:N]
    R2 = R2[N_edge:N, N_edge:N]
    R3 = R3[N_edge:N, N_edge:N]
    NB = N - N_edge
    I = np.eye(NB)
    
    Rtotal2 = np.abs(R1)**2 + np.abs(T2)**2 @ np.linalg.inv(I - np.abs(R3)**2 @ np.abs(R2)**2) @ np.abs(R3)**2 @ np.abs(T1)
    Ttotal2 = np.abs(T3)**2 @ np.linalg.inv(I - np.abs(R2)**2 @ np.abs(R3)**2) @ np.abs(T1)**2
    
    return Rtotal2, Ttotal2


##############################################################
# Boson state generation and perturbative expansion functions


def combination(m, n):
    """Binomial coefficient C(m,n)"""
    return int(np.math.factorial(m) / (np.math.factorial(n) * np.math.factorial(m - n)))


def generate_boson_states(n, m):
    """
    Generate all possible boson occupation number states.
    
    Parameters:
    -----------
    n : int
        Total number of bosons
    m : int
        Number of modes
        
    Returns:
    --------
    states : ndarray
        Array of occupation number configurations
    """
    states = []
    current_state = np.zeros(m, dtype=np.int32)

    def backtrack(index, remaining):
        if index == m:
            if remaining == 0:
                states.append(current_state.copy())
            return
        for i in range(remaining + 1):
            current_state[index] = i
            backtrack(index + 1, remaining - i)

    backtrack(0, n)
    return np.array(states)


def permute_unique(a):
    """
    Generate all unique permutations given occupation numbers.
    
    Parameters:
    -----------
    a : list of int
        Occupation numbers for each mode
        
    Returns:
    --------
    result : ndarray
        Array of all unique permutations
    """
    def backtrack(path, counter):
        if len(path) == len_total:
            result.append(tuple(path))
            return
        for color in counter:
            if counter[color] > 0:
                path.append(color)
                counter[color] -= 1
                backtrack(path, counter)
                path.pop()
                counter[color] += 1

    len_total = sum(a)
    result = []
    ball_counter = {i: a[i] for i in range(len(a)) if a[i] > 0}
    backtrack([], ball_counter)
    return np.array(result)


@jit(nopython=True)
def cal_path(perm1, perm2, R1, T1, R2, T2, R3, T3, alpha, beta, currrent_order):
    """
    Calculate amplitude for a specific scattering path.
    
    Parameters:
    -----------
    perm1, perm2 : Permutations defining the path
    R1, T1, R2, T2, R3, T3 : Scattering matrices
    alpha, beta : Channel indices
    currrent_order : Perturbation order
    
    Returns:
    --------
    result : complex
        Path amplitude
    """
    result = T1[perm1[0], beta]
    for i in range(currrent_order):
        result = result * R3[perm2[i], perm1[i]]
        result = result * R2[perm1[i+1], perm2[i]]
    result = result * T3[alpha, perm1[currrent_order]]
    return result


@jit(nopython=True)
def single_frq_perm(perm1, perm2, R1, T1, R2, T2, R3, T3, currrent_order, N):
    """
    Compute transmission matrix for single frequency configuration.
    """
    T_single_frq = np.zeros((N, N), dtype=np.complex128)
    for alpha in range(N):
        for beta in range(N):
            T_single_frq[alpha, beta] += cal_path(perm1, perm2, R1, T1, R2, T2, R3, T3, alpha, beta, currrent_order)
    return T_single_frq


def exact_expansion(N, N_edge, R1, T1, R2, T2, R3, T3, order=2):
    """
    Exact perturbative expansion of transmission to given order.
    Accounts for all scattering paths up to specified order.
    
    Parameters:
    -----------
    N : int
        System dimension
    N_edge : int
        Number of edge states
    R1, T1, R2, T2, R3, T3 : Scattering matrices
    order : int
        Maximum perturbation order
        
    Returns:
    --------
    Ttotal2 : ndarray
        |T|² including all paths up to given order
    """
    T1 = T1[N_edge:N, :]
    T2 = T2[:, N_edge:N]
    T3 = T3[:, N_edge:N]
    R2 = R2[N_edge:N, N_edge:N]
    R3 = R3[N_edge:N, N_edge:N]
    NB = N - N_edge

    Ttotal2 = (np.abs(T3)**2) @ (np.abs(T1)**2)
    
    for n in range(1, order + 1):
        states1 = generate_boson_states(n + 1, NB)
        states2 = generate_boson_states(n, NB)
        
        for i in range(combination(NB + n, NB - 1)):
            state1 = states1[i, :]
            perms1 = permute_unique(state1)
            
            for j in range(combination(NB + n - 1, NB - 1)):
                state2 = states2[j, :]
                perms2 = permute_unique(state2)
                T_single_frq = np.zeros((np.size(T3, 0), np.size(T1, 1)), dtype=np.complex128)
                
                for k in range(np.size(perms1, 0)):
                    perm1 = perms1[k, :]
                    for l in range(np.size(perms2, 0)):
                        perm2 = perms2[l, :]
                        add = single_frq_perm(perm1, perm2, R1, T1, R2, T2, R3, T3, n, N)
                        T_single_frq += add
                        
                Ttotal2 += np.abs(T_single_frq)**2
                
    return Ttotal2


def shift_up(N):
    """
    Create shift-up operator matrix (circular shift in frequency space).
    
    Parameters:
    -----------
    N : int
        Half dimension
        
    Returns:
    --------
    matrix : ndarray
        Shift operator (2N × 2N)
    """
    matrix = np.zeros((2*N, 2*N))

    # Fill upper-left block (position shift)
    for i in range(N - 1):
        matrix[i + 1, i] = 1
    matrix[0, N - 1] = 1

    # Fill lower-right block (momentum shift)
    for i in range(N - 1):
        matrix[i + N + 1, i + N] = 1
    matrix[N, 2*N - 1] = 1

    return matrix


def time_evo(t_mesh, dt, N, H_list, endpoint=False):
    """
    Time evolution using split-operator method.
    
    Parameters:
    -----------
    t_mesh : int
        Number of time steps
    dt : float
        Time step size
    N : int
        Hilbert space dimension
    H_list : list of ndarray
        Time-dependent Hamiltonian at each time step
    endpoint : bool
        Whether to include endpoint in H_list
        
    Returns:
    --------
    U_list : ndarray
        Evolution operators at each time step
    """
    U = np.eye(N)
    U_list = np.zeros((t_mesh, N, N), dtype=np.complex128)
    
    if endpoint == False:
        half_simalU_list = np.zeros(t_mesh, N, N, dtype=np.complex128)
        for i in range(t_mesh):
            H = H_list[i]
            H_eigenvalues, H_eigenvectors = np.linalg.eigh(H)
            half_simalU_list[i, :, :] = np.dot(H_eigenvectors, 
                                              np.dot(np.diag(np.exp(-1j * H_eigenvalues * 0.5 * dt)), 
                                                    H_eigenvectors.conjugate().T))
        for i in range(t_mesh):
            U_list[i, :, :] = half_simalU_list[(i + 1) % t_mesh, :, :] @ half_simalU_list[i, :, :] @ U 
    else:
        half_simalU_list = np.zeros((t_mesh + 1, N, N), dtype=np.complex128)
        for i in range(t_mesh + 1):
            H = H_list[i]
            H_eigenvalues, H_eigenvectors = np.linalg.eigh(H)
            half_simalU_list[i, :, :] = np.dot(H_eigenvectors, 
                                              np.dot(np.diag(np.exp(-1j * H_eigenvalues * 0.5 * dt)), 
                                                    H_eigenvectors.conjugate().T))
        for i in range(t_mesh):
            U_list[i, :, :] = half_simalU_list[i + 1, :, :] @ half_simalU_list[i, :, :] @ U

    return U_list


def down_N(N, q):
    """
    Create downshift operator in frequency space.
    Parameters:
    -----------
    N : int
        Dimension
    q : int
        Shift amount
        
    Returns:
    --------
    down : ndarray
        Downshift operator matrix
    """
    down = np.zeros((N, N))
    for i in range(N):
        if i + q < N:
            down[i, i + q] = 1
    return down


def cut(half_N_origin, half_N_target):
    """
    Create projection operator for frequency cutoff.
    
    Parameters:
    -----------
    half_N_origin : int
        Original half-width
    half_N_target : int
        Target half-width
        
    Returns:
    --------
    cut : ndarray
        Projection matrix
    """
    N = 2 * half_N_origin + 1
    cut = np.zeros((N, N))
    for i in range(N):
        if i >= half_N_origin - half_N_target and i < half_N_origin + half_N_target + 1:
            cut[i, i] = 1
    return cut


def plot_matrix_modulus_square(matrix, title="Matrix Element Modulus Square", cmap='viridis'):
    """
    Plot heatmap of matrix element modulus squared.
    
    Parameters:
    -----------
    matrix : ndarray
        Input matrix (can be complex or real)
    title : str
        Plot title
    cmap : str
        Colormap name
    """
    # Compute modulus squared of matrix elements
    modulus_square = np.abs(matrix)**2
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    cax = ax.matshow(modulus_square, cmap=cmap)
    
    # Add colorbar
    fig.colorbar(cax, label='|Element|²')
    
    # Set title and axes
    ax.set_title(title)
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    
    # Disable grid
    ax.grid(False)
    
    # Display
    plt.show()
