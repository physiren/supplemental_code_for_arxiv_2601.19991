import os
# Must set environment variables before importing any library that might use OpenMP
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
from scipy.linalg import eig
from numba import jit, complex128, float64, int64, njit
import matplotlib.pyplot as plt
import scipy as sp
import warnings
import pickle
import concurrent.futures
from tqdm import tqdm 
import numba as nb

# Configure numba to use single-threading
nb.config.THREADING_LAYER = 'omp'
nb.config.NUMBA_NUM_THREADS = 1

@jit(nopython=True)
def symplectic_inner_product(Y1, Y2, N):
    """
    Compute the symplectic inner product Y1^† J Y2, utilizing the special structure of J matrix
    
    Parameters:
    Y1: 1D numpy array, first vector
    Y2: 1D numpy array, second vector
    N: int, dimension of the original matrix
    
    Returns:
    Symplectic inner product value (complex)
    """
    # Split vectors into upper and lower parts
    Y1_up = Y1[:N]
    Y1_down = Y1[N:]
    Y2_up = Y2[:N]
    Y2_down = Y2[N:]
    
    # Compute symplectic inner product: Y1_up^† Y2_down - Y1_down^† Y2_up
    inner_prod = 0.0 + 0.0j
    for i in range(N):
        inner_prod += np.conj(Y1_up[i]) * Y2_down[i] - np.conj(Y1_down[i]) * Y2_up[i]
    
    return inner_prod



@jit(nopython=True)
def symplectic_conjugate(v, N):
    """
    Compute v^† J, utilizing the special structure of J matrix
    
    Parameters:
    v: 1D numpy array, input vector
    N: int, dimension of the original matrix
    
    Returns:
    v^† J (row vector)
    """
    # Split vector into upper and lower parts
    v_up = v[:N]
    v_down = v[N:]
    
    # Compute v^† J = [-v_down^†, v_up^†]
    result = np.zeros(2*N, dtype=np.complex128)
    for i in range(N):
        result[i] = -np.conj(v_down[i])  # Upper part: -v_down^†
        result[i+N] = np.conj(v_up[i])   # Lower part: v_up^†
    
    return result



@jit(nopython=True)
def build_generator_matrix_numba(epsilon, V, E_gate, omega, n_bottom, n_top,movedirect='right'):
    """
    Numba-optimized function to build the generator matrix for Floquet matrix
    
    Parameters:
    epsilon: float, energy parameter
    V: float, potential strength
    omega: float, frequency
    n_bottom: int, lowest energy level index
    n_top: int, highest energy level index
    
    Returns:
    G: 2D numpy array, generator matrix for Floquet matrix
    N: int, dimension of the original matrix
    """
    n_values = np.arange(n_bottom, n_top + 1)
    N = len(n_values)
    
    # Pre-allocate matrix
    G = np.zeros((2*N, 2*N), dtype=np.complex128)
    
    # Build B matrix part (diagonal matrix)
    for i in range(N):
        if movedirect=='right':
            G[i, i] = -1j * 2 * np.pi * n_values[i]  # -iB top-left
            G[i+N, i+N] = -1j * 2 * np.pi * n_values[i]  # -iB bottom-right
        elif movedirect=='left':
            G[i, i] = 1j * 2 * np.pi * n_values[i]  # iB top-left
            G[i+N, i+N] = 1j * 2 * np.pi * n_values[i]  # iB bottom-right
        
    
    # Build identity matrix part
    for i in range(N):
        G[i, i+N] = 1.0  # I top-right
    
    # Build M matrix part (tridiagonal matrix)
    # Main diagonal: - (epsilon + n*omega)
    for i in range(N):
        G[i+N, i] = -(epsilon + n_values[i] * omega)+E_gate
    
    # Sub-diagonal: V/2
    for i in range(N-1):
        G[i+N, i+1] = V/2
        G[i+N+1, i] = V/2
    
    return G, N



def build_generator_matrix(epsilon, V,E_gate, omega, n_bottom, n_top,movedirect='right'):
    """
    Wrapper function calling the Numba-optimized version
    """
    return build_generator_matrix_numba(epsilon, V,E_gate, omega, n_bottom, n_top,movedirect)



@jit(nopython=True)
def normalize_eigenvectors(eigenvectors, N):
    """
    Numba-optimized function to normalize eigenvectors
    """
    for i in range(eigenvectors.shape[1]):
        norm = 0.0
        for j in range(N):
            norm += np.abs(eigenvectors[j, i])**2
        norm = np.sqrt(norm)
        
        for j in range(eigenvectors.shape[0]):
            eigenvectors[j, i] /= norm
    
    return eigenvectors



@jit(nopython=True)
def compute_j_values(zero_eigenvectors, N):
    """
    J-value computation function optimized using symplectic inner product
    """
    j_values = np.zeros(zero_eigenvectors.shape[1], dtype=np.float64)
    
    for i in range(zero_eigenvectors.shape[1]):
        Y = zero_eigenvectors[:, i]
        # Use symplectic inner product to compute j = -i * Y^† J Y
        j_val = -1j * symplectic_inner_product(Y, Y, N)
        j_values[i] = np.real(j_val)
    
    return j_values



def diagonalize_and_classify(G, N, tol=1e-8):
    """
    Diagonalize matrix and classify results
    
    Parameters:
    G: 2D numpy array, matrix to be diagonalized
    N: int, dimension of the original matrix
    tol: float, tolerance for judging whether real part of eigenvalue is 0
    
    Returns:
    result_dict: dict, dictionary containing classification results
    """
    # Step 1: Diagonalization
    eigenvalues, eigenvectors = eig(G)
    
    # Step 2: Special normalization (normalize according to the first half of the vector)
    eigenvectors = normalize_eigenvectors(eigenvectors, N)
    
    # Step 3: Classification
    # Separate eigenvalues with real part less than 0 and greater than 0
    real_parts = np.real(eigenvalues)
    
    # Less than 0 part
    neg_mask = real_parts < -tol
    neg_eigenvalues = eigenvalues[neg_mask]
    neg_eigenvectors = eigenvectors[:, neg_mask]
    
    # Sort negative real part eigenvalues (from small to large)
    neg_sort_indices = np.argsort(np.real(neg_eigenvalues))
    K_plus_decay = neg_eigenvalues[neg_sort_indices]
    A_plus_decay = neg_eigenvectors[:, neg_sort_indices]
    
    # Greater than 0 part
    pos_mask = real_parts > tol
    pos_eigenvalues = eigenvalues[pos_mask]
    pos_eigenvectors = eigenvectors[:, pos_mask]
    
    # Sort positive real part eigenvalues (from large to small, reverse order)
    pos_sort_indices = np.argsort(-np.real(pos_eigenvalues))
    K_minus_decay = pos_eigenvalues[pos_sort_indices]
    A_minus_decay = pos_eigenvectors[:, pos_sort_indices]
    
    # Real part close to 0 part
    zero_mask = np.abs(real_parts) <= tol
    zero_eigenvalues = eigenvalues[zero_mask]
    zero_eigenvectors = eigenvectors[:, zero_mask]
    
    # Compute j values
    j_values = compute_j_values(zero_eigenvectors, N)
    
    # Separate j>0 and j<0 parts
    j_pos_mask = j_values > tol
    j_neg_mask = j_values < -tol
    
    # j>0 part, sort from small to large
    j_pos_values = j_values[j_pos_mask]
    j_pos_sort_indices = np.argsort(j_pos_values)
    
    K_plus_bloch = zero_eigenvalues[j_pos_mask][j_pos_sort_indices]
    A_plus_bloch = zero_eigenvectors[:, j_pos_mask][:, j_pos_sort_indices]
    
    # j<0 part, reverse order (from large to small)
    j_neg_values = j_values[j_neg_mask]
    j_neg_sort_indices = np.argsort(-j_neg_values)  # Reverse order sorting
    
    K_minus_bloch = zero_eigenvalues[j_neg_mask][j_neg_sort_indices]
    A_minus_bloch = zero_eigenvectors[:, j_neg_mask][:, j_neg_sort_indices]
    
    # Return result dictionary
    result_dict = {
        'K_plus_decay': K_plus_decay,
        'A_plus_decay': A_plus_decay,
        'K_minus_decay': K_minus_decay,
        'A_minus_decay': A_minus_decay,
        'K_plus_bloch': K_plus_bloch,
        'A_plus_bloch': A_plus_bloch,
        'K_minus_bloch': K_minus_bloch,
        'A_minus_bloch': A_minus_bloch
    }
    
    return result_dict




def compute_dual_vectors(A_decay_1, A_decay_2, A_bloch, N, is_bloch=True):
    """
    Dual vector computation function optimized using symplectic inner product
    
    Parameters:
    A_decay_1: 2D numpy array, decay part matrix 1
    A_decay_2: 2D numpy array, decay part matrix 2
    A_bloch: 2D numpy array, bloch part matrix
    N: int, dimension of the original matrix
    is_bloch: bool, whether to compute bloch part
    
    Returns:
    A_sharp_decay: 2D numpy array, dual decay part matrix
    A_sharp_bloch: 2D numpy array, dual bloch part matrix
    """
    if is_bloch:
        # Compute bloch part
        n_bloch = A_bloch.shape[1]
        A_sharp_bloch = np.zeros((n_bloch, 2*N), dtype=np.complex128)
        
        for i in range(n_bloch):
            v = A_bloch[:, i]
            
            # Use symplectic inner product to compute denominator: v^† J v
            denominator = symplectic_inner_product(v, v, N)
            
            # Use symplectic conjugate to compute numerator: v^† J
            v_dag_J = symplectic_conjugate(v, N)
            
            # Compute dual vector
            for k in range(2*N):
                A_sharp_bloch[i, k] = v_dag_J[k] / denominator
        
        return None, A_sharp_bloch
    else:
        # Compute decay part
        n_decay = A_decay_1.shape[1]
        A_sharp_decay = np.zeros((n_decay, 2*N), dtype=np.complex128)
        
        for i in range(n_decay):
            v1 = A_decay_1[:, i]
            v2 = A_decay_2[:, i]
            
            # Use symplectic inner product to compute denominator: v2^† J v1
            denominator = symplectic_inner_product(v2, v1, N)
            
            # Use symplectic conjugate to compute numerator: v2^† J
            v2_dag_J = symplectic_conjugate(v2, N)
            
            # Compute dual vector
            for k in range(2*N):
                A_sharp_decay[i, k] = v2_dag_J[k] / denominator
        
        return A_sharp_decay, None


def build_dual_space(result_dict, N):
    """
    Build dual space
    
    Parameters:
    result_dict: dict, result dictionary obtained from diagonalize_and_classify function
    N: int, dimension of the original matrix
    
    Returns:
    A_plus: 2D numpy array, combined A_plus matrix
    A_minus: 2D numpy array, combined A_minus matrix
    A_sharp_plus: 2D numpy array, dual A_sharp_plus matrix
    A_sharp_minus: 2D numpy array, dual A_sharp_minus matrix
    """
    # Extract each part from result dictionary
    A_plus_decay = result_dict['A_plus_decay']
    A_minus_decay = result_dict['A_minus_decay']
    A_plus_bloch = result_dict['A_plus_bloch']
    A_minus_bloch = result_dict['A_minus_bloch']
    
    # Check if dimensions match
    if A_plus_decay.shape[1] != A_minus_decay.shape[1]:
        print(f"Warning: Decay part dimensions do not match: A_plus_decay {A_plus_decay.shape}, A_minus_decay {A_minus_decay.shape}")
    
    if A_plus_bloch.shape[1] != A_minus_bloch.shape[1]:
        print(f"Warning: Bloch part dimensions do not match: A_plus_bloch {A_plus_bloch.shape}, A_minus_bloch {A_minus_bloch.shape}")
    
    # Compute each dual part
    A_sharp_plus_decay, _ = compute_dual_vectors(A_plus_decay, A_minus_decay, None, N, is_bloch=False)
    A_sharp_plus_bloch = compute_dual_vectors(None, None, A_plus_bloch, N, is_bloch=True)[1]
    A_sharp_minus_decay, _ = compute_dual_vectors(A_minus_decay, A_plus_decay, None, N, is_bloch=False)
    A_sharp_minus_bloch = compute_dual_vectors(None, None, A_minus_bloch, N, is_bloch=True)[1]
    
    # Combine A_plus and A_minus
    A_plus = np.hstack([A_plus_decay, A_plus_bloch])
    A_minus = np.hstack([A_minus_decay, A_minus_bloch])
    
    # Combine A_sharp_plus and A_sharp_minus
    A_sharp_plus = np.vstack([A_sharp_plus_decay, A_sharp_plus_bloch])
    A_sharp_minus = np.vstack([A_sharp_minus_decay, A_sharp_minus_bloch])
    
    return A_plus, A_minus, A_sharp_plus, A_sharp_minus

@jit(nopython=True)
def gen_free_space(n_bottom,n_top,epsilon,omega,E_gate=0):
    N=n_top-n_bottom+1
    Fp=np.zeros((2*N,N),dtype=np.complex128)
    Fn=np.zeros((2*N,N),dtype=np.complex128)
    Fp_sharp=np.zeros((N,2*N),dtype=np.complex128)
    Fn_sharp=np.zeros((N,2*N),dtype=np.complex128)
    
    for i in range(N):
        n=i+n_bottom
        Ek=epsilon+n*omega-E_gate
        if Ek>=0:
            k=1j*np.sqrt(Ek)
            Fp[i,i]=1/np.sqrt(np.abs(k))
            Fp[i+N,i]=k/np.sqrt(np.abs(k))
            Fp_sharp[i,i]=k/np.sqrt(np.abs(k))*(-0.5j)
            Fp_sharp[i,i+N]=1/np.sqrt(np.abs(k))*(-0.5j)
            Fn[i,i]=1/np.sqrt(np.abs(k))
            Fn[i+N,i]=-k/np.sqrt(np.abs(k))
            Fn_sharp[i,i]=-k/np.sqrt(np.abs(k))*(0.5j)
            Fn_sharp[i,i+N]=1/np.sqrt(np.abs(k))*(0.5j)
        else:
            k=-np.sqrt(-Ek)
            Fp[i,i]=1/np.sqrt(np.abs(k))
            Fp[i+N,i]=k/np.sqrt(np.abs(k))
            Fp_sharp[i,i]=k/np.sqrt(np.abs(k))*(-0.5)
            Fp_sharp[i,i+N]=1/np.sqrt(np.abs(k))*(-0.5)
            Fn[i,i]=1/np.sqrt(np.abs(k))
            Fn[i+N,i]=-k/np.sqrt(np.abs(k))
            Fn_sharp[i,i]=-k/np.sqrt(np.abs(k))*(0.5)
            Fn_sharp[i,i+N]=1/np.sqrt(np.abs(k))*(0.5)

    return Fp, Fn, Fp_sharp, Fn_sharp


def semi_inf(A_plus, A_minus, A_sharp_plus, A_sharp_minus,B_plus, B_minus, B_sharp_plus, B_sharp_minus,side):
    if side=='left':
        TL=sp.linalg.inv(np.dot(A_sharp_plus,B_plus))
        RL=np.dot(np.dot(A_sharp_minus,B_plus),TL)
        return TL, RL
    elif side=='right':
        TR=sp.linalg.inv(np.dot(B_sharp_minus,A_minus))
        RR=np.dot(np.dot(B_sharp_plus,A_minus),TR)
        return TR,RR



def Fermi_Dirac(chem_potential,T_E,E):
    return 1/(np.exp((E-chem_potential)/T_E)+1)

@njit(nogil=True, cache=True, fastmath=True)
def trans_MC(T1, R2, R3, T2, N_MC=1000):
    # Convert input arrays to C-contiguous if they are not
    T1 = np.ascontiguousarray(T1)
    R2 = np.ascontiguousarray(R2)
    R3 = np.ascontiguousarray(R3)
    T2 = np.ascontiguousarray(T2)
    
    N_bloch_1side = T1.shape[0]
    N = T1.shape[1]
    I = np.eye(N_bloch_1side)  # This is C-contiguous by default
    T_sqr = np.zeros((N, N), dtype=np.float64)
    
    for time in range(N_MC):
        phases1 = np.random.uniform(0, 2*np.pi, N_bloch_1side)
        phases2 = np.random.uniform(0, 2*np.pi, N_bloch_1side)
        
        T1_randphase = np.zeros_like(T1, dtype=np.complex128)
        R2_randphase = np.zeros_like(R2, dtype=np.complex128)
        R3_randphase = np.zeros_like(R3, dtype=np.complex128)
        
        for i in range(N_bloch_1side):
            phase_factor1 = np.exp(1j * phases1[i])
            phase_factor2 = np.exp(1j * phases2[i])
            for j in range(N):
                T1_randphase[i, j] = phase_factor1 * T1[i, j]
            for j in range(N_bloch_1side):
                R3_randphase[i, j] = phase_factor1 * R3[i, j]
                R2_randphase[i, j] = phase_factor2 * R2[i, j]
        
        inv_mat = np.linalg.inv(I - R3_randphase @ R2_randphase)
        T = T2 @ inv_mat @ T1_randphase
        T_sqr += np.abs(T)**2 / N_MC
        
    return T_sqr



@njit(nogil=True, cache=True, fastmath=True)
def trans_exact(T1, R2, R3, T2,K_plus,K_minus,L):
    # Convert input arrays to C-contiguous if they are not
    T1 = np.ascontiguousarray(T1)
    R2 = np.ascontiguousarray(R2)
    R3 = np.ascontiguousarray(R3)
    T2 = np.ascontiguousarray(T2)
    
    N = T1.shape[1]
    I = np.eye(N)  # This is C-contiguous by default
    
    T1_withphase = np.zeros_like(T1, dtype=np.complex128)
    R2_withphase = np.zeros_like(R2, dtype=np.complex128)
    R3_withphase = np.zeros_like(R3, dtype=np.complex128)
        
    for i in range(N):
        phase_plus = np.exp(L*K_plus[i])
        phase_minus = np.exp(-L*K_minus[i])
        for j in range(N):
            T1_withphase[i, j] = phase_plus * T1[i, j]
            R3_withphase[i, j] = phase_plus * R3[i, j]
            R2_withphase[i, j] = phase_minus * R2[i, j]
    
    inv_mat = np.linalg.inv(I - R3_withphase @ R2_withphase)
    T = T2 @ inv_mat @ T1_withphase
    T_sqr = np.abs(T)**2
    
    return T_sqr




def S_matrix_merge(T1L,T1R,R1L,R1R,T2L,T2R,R2L,R2R):
    N=np.size(T1L,0)

    L_inv_term=sp.linalg.inv(np.eye(N)-np.dot(R1R,R2L))
    TL=T2L@L_inv_term@T1L
    RL=R1L+T1R@R2L@L_inv_term@T1L

    R_inv_term=sp.linalg.inv(np.eye(N)-np.dot(R2L,R1R))
    TR=T1R@R_inv_term@T2R
    RR=R2R+T2L@R1R@R_inv_term@T2R

    return TL,TR,RL,RR


def Airy_trans(n_bottom,n_top,epsilon,omega,L_boundary,E_gate):
    N=n_top-n_bottom+1
    T_F2G=np.zeros((N,N),dtype=np.complex128)
    R_F2F=np.zeros((N,N),dtype=np.complex128)
    T_G2F=np.zeros((N,N),dtype=np.complex128)
    R_G2G=np.zeros((N,N),dtype=np.complex128)

    for index in range(N):
        n=n_bottom+index
        E=epsilon+n*omega
        # Compute wave number
        if E >= 0:
            k= np.sqrt(E)
        else:
            k= 1j * np.sqrt(-E)
        
        
        # Compute Airy function parameters
        c = (E_gate / L_boundary) ** (1/3)  # Scaling factor
        x0 = E * L_boundary / E_gate  # Offset
        
        # Compute ξ values at boundary points
        xi0 = c * (0 - x0)  # ξ at x=0
        xiL = c * (L_boundary - x0)  # ξ at x=L_boundary
        
        # Compute Airy functions and their derivatives at boundary points
        Ai0, Aip0, Bi0, Bip0 = sp.special.airy(xi0)
        AiL, AipL, BiL, BipL = sp.special.airy(xiL)

        Bi0=Bi0/BiL
        Bip0=Bip0/BiL
        BipL=BipL/BiL
        BiL=1

        
        Aip0=Aip0/Ai0
        AipL=AipL/Ai0
        AiL=AiL/Ai0
        Ai0=1
        
        # Compute scattering coefficients from left to right
        # Set up matrix equation: M * [A, B, r, t]^T = v
        M_left = np.array([
            [Ai0, Bi0, -1, 0],
            [c * Aip0, c * Bip0, 1j * k, 0],
            [AiL, BiL, 0, -1],
            [c * AipL, c * BipL, 0, -1j * k]
        ], dtype=complex)
        
        v_left = np.array([1, 1j * k, 0, 0], dtype=complex)
        
        # Solve linear system
        solution_left = np.linalg.solve(M_left, v_left)
        A, B, r, t = solution_left
        
        # Compute scattering coefficients from right to left
        # Set up matrix equation: M_prime * [A', B', r', t']^T = v_prime
        M_right = np.array([
            [AiL, BiL, -1, 0],
            [c * AipL, c * BipL, -1j * k, 0],
            [Ai0, Bi0, 0, -1],
            [c * Aip0, c * Bip0, 0, 1j * k]
        ], dtype=complex)
        
        v_right = np.array([1, -1j * k, 0, 0], dtype=complex)
        
        # Solve linear system
        solution_right = np.linalg.solve(M_right, v_right)
        A_prime, B_prime, r_prime, t_prime = solution_right
        T_F2G[index,index]=t
        R_F2F[index,index]=r
        T_G2F[index,index]=t_prime
        R_G2G[index,index]=r_prime
    
    return T_F2G,T_G2F,R_F2F,R_G2G




def LB_experiment_single_epsilon(V,omega,epsilon,n_bottom,n_top,E_gate,L_period,linear_length):
    F_plus,F_minus,F_sharp_plus,F_sharp_minus=gen_free_space(n_bottom,n_top,epsilon,omega,0)
    G,N=build_generator_matrix(epsilon,V,E_gate,omega,n_bottom,n_top)


    diag_clsf_result = diagonalize_and_classify(G, N)

    

    N_decay_1side=np.size(diag_clsf_result['A_plus_decay'],1)
    A_plus,A_minus,A_sharp_plus,A_sharp_minus=build_dual_space(diag_clsf_result,N)
    K_plus_decay=diag_clsf_result['K_plus_decay']
    K_plus_bloch=diag_clsf_result['K_plus_bloch']
    K_minus_decay=diag_clsf_result['K_minus_decay']
    K_minus_bloch=diag_clsf_result['K_minus_bloch']
    K_plus=np.hstack([K_plus_decay, K_plus_bloch])
    K_minus=np.hstack([K_minus_decay, K_minus_bloch])



    # Calculate matrices needed for left incidence
    TL_LS,RL_LS=semi_inf(F_plus, F_minus, F_sharp_plus, F_sharp_minus,A_plus, A_minus, A_sharp_plus, A_sharp_minus,'left')
    TR_LS,RR_LS=semi_inf(F_plus, F_minus, F_sharp_plus, F_sharp_minus,A_plus, A_minus, A_sharp_plus, A_sharp_minus,'right')

    TL_RS,RL_RS=semi_inf(A_plus, A_minus, A_sharp_plus, A_sharp_minus,F_plus, F_minus, F_sharp_plus, F_sharp_minus,'left')
    TR_RS,RR_RS=semi_inf(A_plus, A_minus, A_sharp_plus, A_sharp_minus,F_plus, F_minus, F_sharp_plus, F_sharp_minus,'right')

    # Ramp
    if linear_length==0:
        pass
    else:
        T_F2G,T_G2F,R_F2F,R_G2G=Airy_trans(n_bottom,n_top,epsilon,omega,linear_length,E_gate)

        TL_LS,TR_LS,RL_LS,RR_LS=S_matrix_merge(T_F2G,T_G2F,R_F2F,R_G2G,TL_LS,TR_LS,RL_LS,RR_LS)
        TL_RS,TR_RS,RL_RS,RR_RS=S_matrix_merge(TL_RS,TR_RS,RL_RS,RR_RS,T_G2F,T_F2G,R_G2G,R_F2F)

    if L_period==0:
        # Trim matrix, keep only Bloch wave transmission and reflection
        TL_LS=TL_LS[N_decay_1side:,:]
        TR_LS=TR_LS[:,N_decay_1side:]
        RR_LS=RR_LS[N_decay_1side:,N_decay_1side:]

        TL_RS=TL_RS[:,N_decay_1side:]
        RL_RS=RL_RS[N_decay_1side:,N_decay_1side:]
        TR_RS=TR_RS[N_decay_1side:,:]
        T_sqr_L=trans_MC(TL_LS,RL_RS,RR_LS,TL_RS)
        T_sqr_R=trans_MC(TR_RS,RR_LS,RL_RS,TR_LS)
    else:
        T_sqr_L=trans_exact(TL_LS,RL_RS,RR_LS,TL_RS,K_plus,K_minus,L_period)
        T_sqr_R=trans_exact(TR_RS,RR_LS,RL_RS,TR_LS,-K_minus,-K_plus,L_period)

    return T_sqr_L,T_sqr_R



def LB(V,E_gate,chem_potential,E_T, E_safe,omega,E_mesh,L_period,linear_length):
    n_top=int((E_gate+E_safe)//omega)
    n_bottom=int((E_gate-E_safe)//omega)
    N=n_top-n_bottom+1
    # Energy grid for one period [-ω/2, ω/2]
    epsilon_list = np.linspace(-omega/2 + omega/E_mesh/2, 
                         omega/2 - omega/E_mesh/2, 
                         E_mesh, endpoint=True)

    # Extended energy grid for all frequency components
    E_list = np.linspace(-omega/2 + omega/E_mesh/2 +n_bottom*omega,
                         omega/2 - omega/E_mesh/2 + n_top*omega,
                         E_mesh*N, endpoint=True)
    
    current_TL_list = np.zeros(E_mesh * N)
    current_TR_list = np.zeros(E_mesh * N)

    for i, epsilon in enumerate(epsilon_list):
        T_sqr_L, T_sqr_R = LB_experiment_single_epsilon(V,omega,epsilon,n_bottom,n_top,E_gate,L_period,linear_length)
        
        metric = np.linspace(omega*n_bottom, omega*n_top, N, endpoint=True) + epsilon
        metric = (np.sign(metric) > 0).astype(int)


        

        tl_results = np.zeros(N)
        tr_results = np.zeros(N)


        for j in range(N):
            tl_results[j] = np.dot(T_sqr_L[:,j], metric)
            tr_results[j] = np.dot(T_sqr_R[:,j], metric)

        current_TL_list[i::E_mesh] = tl_results
        current_TR_list[i::E_mesh] = tr_results

    if E_T==0:
        mask1 =E_list < chem_potential
        mask2= E_list > 0
        mask=mask1*mask2
        diff = current_TL_list - current_TR_list
        current= np.sum(diff[mask])/E_mesh
    else:
        f=Fermi_Dirac(chem_potential,E_T,E_list)
        diff = current_TL_list - current_TR_list
        mask=E_list>0
        diff=diff*mask
        current= np.sum(diff*f)/E_mesh
    

    return current,E_list,current_TL_list,current_TR_list



def unit_trans(d_um,f_MHz,chem_potential_eV,T_K,V_eV,ita=0.023):
    h=6.62607015e-34
    q=1.6e-19
    K_B=1.38064e-23
    me=(9.109e-31) * ita
    d_m=d_um*1e-6
    E_band=h**2/(8*d_m**2*me)

    V=V_eV*q/E_band * np.pi**2
    chem_potential=chem_potential_eV*q/E_band * np.pi**2
    E_T=K_B*T_K/E_band*np.pi**2
    omega=h*f_MHz*1e6/E_band * np.pi**2

    print('omega',omega)
    print('V',V)
    print('chem_potential',chem_potential)
    print('V/omega',V/omega)
    print('temprature',E_T)



    return omega,V,chem_potential,E_T



def phase_diagram(E_gate_bottom, E_gate_top, E_gate_mesh, V_bottom, V_top, V_mesh, 
                  omega, E_mesh, chem_potential, E_safe, L_period, linear_length):
    """
    Function to plot phase diagram, scanning two-dimensional parameter space and computing current
    
    Parameters:
    E_gate_bottom, E_gate_top: Scan range for gate voltage
    E_gate_mesh: Number of grid points for gate voltage
    V_bottom, V_top: Scan range for bias voltage  
    V_mesh: Number of grid points for bias voltage
    omega: Frequency parameter
    E_mesh: Number of energy grid points
    chem_potential: Chemical potential
    E_safe: Safety energy parameter
    L_period: Period length, default is 10
    
    Returns:
    data_dict: Dictionary containing all plotting data
    """
    
    # Create parameter grids
    E_gate_list = np.linspace(E_gate_bottom, E_gate_top, E_gate_mesh, endpoint=True)
    V_list = np.linspace(V_bottom, V_top, V_mesh, endpoint=True)
    
    # Initialize current matrix
    current_matrix = np.zeros((V_mesh, E_gate_mesh))
    
    # Scan two-dimensional parameter space and compute current
    print("Starting parameter space scan...")
    for i, V in enumerate(V_list):
        for j, E_gate in enumerate(E_gate_list):
            # Call LB function to compute current (assuming LB function is defined)
            current,E_list,current_TL_list,current_TR_list= LB(V, E_gate, chem_potential, E_safe, omega, E_mesh, L_period,linear_length)
            current_matrix[i, j] = current
            
        # Show progress
        if (i+1) % max(1, V_mesh//10) == 0:
            print(f"Progress: {i+1}/{V_mesh} ({100*(i+1)/V_mesh:.1f}%)")
    
    # Create data dictionary
    data_dict = {
        'E_gate_list': E_gate_list,
        'V_list': V_list,
        'current_matrix': current_matrix,
        'parameters': {
            'E_gate_range': (E_gate_bottom, E_gate_top),
            'V_range': (V_bottom, V_top),
            'omega': omega,
            'chem_potential': chem_potential,
            'E_safe': E_safe,
            'L_period': L_period
        }
    }
    
    # Save data to disk
    filename = f"phase_diagram_data_{np.random.randint(10000000,99999999)}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Data saved to: {filename}")
    
    return data_dict


def plot_phase_diagram(data_source=None, **kwargs):
    """
    Function to plot phase diagram, supports reading data from file or passing parameters directly
    
    Parameters:
    data_source: Data source, can be filename or data_dict
    **kwargs: If data_source is None, call phase_diagram with these parameters
    """
    
    if data_source is None:
        # Direct calculation mode
        print("Direct calculation mode...")
        data_dict = phase_diagram(**kwargs)
    elif isinstance(data_source, str):
        # Read from file mode
        print(f"Reading data from file: {data_source}")
        with open(data_source, 'rb') as f:
            data_dict = pickle.load(f)
    else:
        # Directly pass data dictionary mode
        data_dict = data_source
    
    # Extract data
    E_gate_list = data_dict['E_gate_list']
    V_list = data_dict['V_list']
    current_matrix = data_dict['current_matrix']
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    im = plt.imshow(current_matrix, extent=[E_gate_list[0], E_gate_list[-1], 
                                           V_list[0], V_list[-1]], 
                   origin='lower', aspect='auto', cmap='hot')
    
    plt.colorbar(im, label='Current')
    plt.xlabel('Gate Voltage (E_gate)')
    plt.ylabel('Bias Voltage (V)')
    plt.title('Phase Diagram: Current vs Gate and Bias Voltage')
    
    # Add contours
    contour_levels = np.linspace(np.min(current_matrix), np.max(current_matrix), 10)
    plt.contour(E_gate_list, V_list, current_matrix, levels=contour_levels, 
                colors='white', alpha=0.5, linewidths=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return data_dict



def phase_diagram_parallel(E_gate_bottom, E_gate_top, E_gate_mesh, V_bottom, V_top, V_mesh, 
                          omega, E_mesh, chem_potential,E_T, E_safe,L_period, linear_length, max_workers=None):
    """
    Parallel version of phase diagram calculation function
    
    Parameters:
    max_workers: Maximum number of worker processes, None means using CPU core count
    """
    
    # Create parameter grids
    E_gate_list = np.linspace(E_gate_bottom, E_gate_top, E_gate_mesh, endpoint=True)
    V_list = np.linspace(V_bottom, V_top, V_mesh, endpoint=True)
    
    # Initialize current matrix
    current_matrix = np.zeros((V_mesh, E_gate_mesh))
    
    # Prepare parameter combinations
    params_list = []
    for i, V in enumerate(V_list):
        for j, E_gate in enumerate(E_gate_list):
            params_list.append((i, j, V, E_gate))
    
    print(f"Starting parallel computation, total {len(params_list)} tasks, using {max_workers} worker processes...")
    
    # Use process pool for parallel computation
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_params = {
            executor.submit(compute_single_point, V, E_gate, chem_potential, E_T, E_safe, omega, E_mesh, L_period,linear_length): (i, j) 
            for i, j, V, E_gate in params_list
        }
        
        # Use tqdm to show progress bar (optional)
        for future in tqdm(concurrent.futures.as_completed(future_to_params), 
                          total=len(params_list), desc="Computation progress"):
            i, j = future_to_params[future]
            try:
                current, E_list, current_TL_list, current_TR_list = future.result()
                current_matrix[i, j] = current
            except Exception as exc:
                print(f'Exception occurred for parameter point ({i}, {j}): {exc}')
                current_matrix[i, j] = np.nan  # Mark error point
    
    # Create data dictionary
    data_dict = {
        'E_gate_list': E_gate_list,
        'V_list': V_list,
        'current_matrix': current_matrix,
        'parameters': {
            'E_gate_range': (E_gate_bottom, E_gate_top),
            'V_range': (V_bottom, V_top),
            'omega': omega,
            'chem_potential': chem_potential,
            'E_safe': E_safe,
            'L_period': L_period
        }
    }
    
    # Save data to disk
    filename = f"phase_diagram_data_{np.random.randint(10000,99999)}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Data saved to: {filename}")
    
    return data_dict

def compute_single_point(V, E_gate, chem_potential,E_T, E_safe, omega, E_mesh, L_period,linear_length):
    """
    Helper function to compute a single parameter point
    """
    current, E_list, current_TL_list, current_TR_list = LB(V, E_gate, chem_potential,E_T, E_safe, omega, E_mesh, L_period,linear_length)
    return current, E_list, current_TL_list, current_TR_list




if __name__ == "__main__":



    d_um=0.16
    f_MHz=25000
    chem_potential_eV=0.010
    V_eV=0.0002
    T_K=2


    

    E_mesh = 64
    omega,V,chem_potential,E_T=unit_trans(d_um,f_MHz,chem_potential_eV,T_K,V_eV)
    chem_potential=70
    
    start1=1
    start2=0
    if start1==1:
        
        params1 = {
        'E_gate_bottom': 100,
        'E_gate_top': 164,
        'E_gate_mesh': 129,
        'V_bottom': 0,
        'V_top': 10,
        'V_mesh': 65,
        'omega': 1.5977624673059374,
        'E_mesh': 32,
        'chem_potential': 154.32495520377188,
        'E_T': 0,
        'E_safe': 64,
        'L_period': 10,
        'linear_length': 0
        }
    

        params2 = {
        'E_gate_bottom': 100,
        'E_gate_top': 164,
        'E_gate_mesh': 129,
        'V_bottom': 0,
        'V_top': 10,
        'V_mesh': 65,
        'omega': 1.5977624673059374,
        'E_mesh': 32,
        'chem_potential': 154.32495520377188,
        'E_T': 1.3316700385,
        'E_safe': 64,
        'L_period': 10,
        'linear_length': 0
        }

        params3 = {
        'E_gate_bottom': 100,
        'E_gate_top': 164,
        'E_gate_mesh': 129,
        'V_bottom': 0,
        'V_top': 10,
        'V_mesh': 65,
        'omega': 1.5977624673059374,
        'E_mesh': 32,
        'chem_potential': 154.32495520377188,
        'E_T': 2.6633400769066955,
        'E_safe': 64,
        'L_period': 10,
        'linear_length': 0
        }


        params4 = {
        'E_gate_bottom': 100,
        'E_gate_top': 164,
        'E_gate_mesh': 129,
        'V_bottom': 0,
        'V_top': 10,
        'V_mesh': 65,
        'omega': 1.5977624673059374,
        'E_mesh': 32,
        'chem_potential': 154.32495520377188,
        'E_T': 0,
        'E_safe': 64,
        'L_period': 10,
        'linear_length': 5/16
        }
    
    
        params5 = {
        'E_gate_bottom': 100,
        'E_gate_top': 164,
        'E_gate_mesh': 129,
        'V_bottom': 0,
        'V_top': 10,
        'V_mesh': 65,
        'omega': 1.5977624673059374,
        'E_mesh': 32,
        'chem_potential': 154.32495520377188,
        'E_T': 1.3316700385,
        'E_safe': 64,
        'L_period': 10,
        'linear_length': 5/16
        }

        params6 = {
        'E_gate_bottom': 100,
        'E_gate_top': 164,
        'E_gate_mesh': 129,
        'V_bottom': 0,
        'V_top': 10,
        'V_mesh': 65,
        'omega': 1.5977624673059374,
        'E_mesh': 32,
        'chem_potential': 154.32495520377188,
        'E_T': 2.6633400769066955,
        'E_safe': 64,
        'L_period': 10,
        'linear_length': 5/16
        }


        params7 = {
        'E_gate_bottom': 100,
        'E_gate_top': 164,
        'E_gate_mesh': 129,
        'V_bottom': 0,
        'V_top': 10,
        'V_mesh': 65,
        'omega': 1.5977624673059374,
        'E_mesh': 32,
        'chem_potential': 154.32495520377188,
        'E_T': 0,
        'E_safe': 64,
        'L_period': 10,
        'linear_length': 5/8
        }
    
    
        params8 = {
        'E_gate_bottom': 100,
        'E_gate_top': 164,
        'E_gate_mesh': 129,
        'V_bottom': 0,
        'V_top': 10,
        'V_mesh': 65,
        'omega': 1.5977624673059374,
        'E_mesh': 32,
        'chem_potential': 154.32495520377188,
        'E_T': 1.3316700385,
        'E_safe': 64,
        'L_period': 10,
        'linear_length': 5/8
        }

        params9 = {
        'E_gate_bottom': 100,
        'E_gate_top': 164,
        'E_gate_mesh': 129,
        'V_bottom': 0,
        'V_top': 10,
        'V_mesh': 65,
        'omega': 1.5977624673059374,
        'E_mesh': 32,
        'chem_potential': 154.32495520377188,
        'E_T': 2.6633400769066955,
        'E_safe': 64,
        'L_period': 10,
        'linear_length': 5/8
        }


      

        

        data = phase_diagram_parallel(**params1, max_workers=24)
        data = phase_diagram_parallel(**params2, max_workers=24)
        data = phase_diagram_parallel(**params3, max_workers=24)
        data = phase_diagram_parallel(**params4, max_workers=24)

        data = phase_diagram_parallel(**params5, max_workers=24)
        data = phase_diagram_parallel(**params6, max_workers=24)
        data = phase_diagram_parallel(**params7, max_workers=24)
        data = phase_diagram_parallel(**params8, max_workers=24)

        data = phase_diagram_parallel(**params9, max_workers=24)



        '''
        #Mode 2: First calculate and save data, then read from file for plotting
        #plot_phase_diagram(data_source='phase_diagram_data_41323.pkl')

        '''
    if start2==1:
        #current,E_list,current_TL_list,current_TR_list=LB(V,90,chem_potential,100,omega,E_mesh,10)
        current,E_list,current_TL_list,current_TR_list=LB(V, 0, chem_potential,E_T, 100, omega, E_mesh, 10,2/3)
        print(current)


        # Create figure
        plt.figure(figsize=(10, 6))
        # Plot left transmission (blue)
        plt.plot(E_list, current_TL_list, 'b-', linewidth=2, label='Left Transmission (TL)')

        # Plot right transmission (red)
        plt.plot(E_list, current_TR_list, 'r-', linewidth=2, label='Right Transmission (TR)')

        # Set chart title and labels
        plt.title('Transmission vs Energy', fontsize=14)
        plt.xlabel('Energy', fontsize=12)
        plt.ylabel('Transmission', fontsize=12)

        # Add legend
        plt.legend()

        # Set grid
        plt.grid(True, alpha=0.3)

        # Set coordinate axis ranges (adjust as needed)
        plt.xlim(min(E_list), max(E_list))
        plt.ylim(0, 1)  # Transmission typically between 0 and 1


        # Show figure
        plt.tight_layout()
        plt.show()