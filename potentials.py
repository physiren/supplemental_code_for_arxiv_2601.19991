"""
Potential functions for Floquet system calculations.
Contains functions to generate M matrix lists for time-periodic potentials.
"""

import numpy as np


def double_wave(mu, V1, V2, shift, nd1, nd2, d, omega, x_mesh, frqcutoff_oneside, E):
    """
    Generate M matrix list for a double cosine potential.
    
    The potential has the form: V(x,t) = V1*cos(2πx/d - nd1*ωt) + V2*cos(2πx/d - nd2*ωt)
    
    Parameters:
    -----------
    mu : float
        Mass parameter
    V1, V2 : float
        Amplitudes of the two cosine potentials
    shift : float
        Energy shift
    nd1, nd2 : int
        Integer multiples of ωt in the cosine arguments
    d : float
        Period of the potential
    omega : float
        Driving frequency
    x_mesh : int
        Number of spatial discretization points
    frqcutoff_oneside : int
        Frequency cutoff on one side (total basis size = 2*frqcutoff_oneside + 1)
    E : float
        Incident energy
    
    Returns:
    --------
    list
        [x_mesh, dx, N, M_list] where:
        - x_mesh: number of spatial points
        - dx: spatial step size
        - N: total basis size
        - M_list: 3D array of shape (x_mesh, N, N) containing M matrices at each spatial point
    """
    # Calculate basis size and spatial parameters
    N = frqcutoff_oneside * 2 + 1
    b = 2 * np.pi / d  # Wavevector
    dx = d / x_mesh  # Spatial step size
    
    # Initialize M matrix list
    M_list = np.zeros((x_mesh, N, N), dtype=np.complex128)
    
    # Construct M matrices at each spatial point
    for i in range(x_mesh):
        x = i * dx  # Spatial coordinate
        
        for j in range(N):
            # Add coupling from nd1 term (if within matrix bounds)
            if 0 <= j + nd1 < N:
                M_list[i, j, j + nd1] += mu * V1 * np.exp(-1j * b * x)
                M_list[i, j + nd1, j] += mu * V1 * np.exp(1j * b * x)
            
            # Add coupling from nd2 term (if within matrix bounds)
            if 0 <= j + nd2 < N:
                M_list[i, j, j + nd2] += mu * V2 * np.exp(-1j * b * x)
                M_list[i, j + nd2, j] += mu * V2 * np.exp(1j * b * x)
            
            # Add diagonal term: kinetic energy + energy shift
            M_list[i, j, j] += 2 * mu * (-E - (j - frqcutoff_oneside) * omega + shift)
    
    return [x_mesh, dx, N, M_list]


def connect_doublewave(mu, V1, V2, shift, nd1, nd2, d, omega, x_mesh, frqcutoff_oneside, E, ratio, side='left'):
    """
    Generate M matrix list for adiabatic connection region.
    
    This function creates a gradually turned-on potential for smooth connection
    between free space and the full potential region.
    
    Parameters:
    -----------
    mu : float
        Mass parameter
    V1, V2 : float
        Amplitudes of the two cosine potentials
    shift : float
        Energy shift
    nd1, nd2 : int
        Integer multiples of ωt in the cosine arguments
    d : float
        Period of the potential
    omega : float
        Driving frequency
    x_mesh : int
        Number of spatial discretization points
    frqcutoff_oneside : int
        Frequency cutoff on one side (total basis size = 2*frqcutoff_oneside + 1)
    E : float
        Incident energy
    ratio : float
        Fractional length of the connection region (0 < ratio <= 1)
    side : str
        'left' or 'right' - which side the connection is on
    
    Returns:
    --------
    list
        [x_mesh, dx, N, M_list] where:
        - x_mesh: number of spatial points
        - dx: spatial step size (positive for left, negative for right)
        - N: total basis size
        - M_list: 3D array of shape (x_mesh, N, N) containing M matrices at each spatial point
    """
    # Calculate basis size and spatial parameters
    N = frqcutoff_oneside * 2 + 1
    b = 2 * np.pi / d  # Wavevector
    
    # Set spatial step direction based on side
    if side == 'left':
        dx = d * ratio / x_mesh  # Positive step for left side
    elif side == 'right':
        dx = -d * ratio / x_mesh  # Negative step for right side
    
    # Initialize M matrix list
    M_list = np.zeros((x_mesh, N, N), dtype=np.complex128)
    
    # Construct M matrices at each spatial point
    for i in range(x_mesh):
        # Linear ramp parameter r from 0 to 1 across the connection region
        r = (i + 0.5) / x_mesh
        
        # Calculate spatial coordinate (centered at connection point)
        if side == 'left':
            x = (i + 0.5) * dx - d * ratio  # Start at -d*ratio, end at 0
        elif side == 'right':
            x = (i + 0.5) * dx + d * ratio  # Start at d*ratio, end at 0
        
        # Determine matrix index (reverse order for right side)
        if side == 'left':
            index = i  # Normal order for left side
        elif side == 'right':
            index = x_mesh - 1 - i  # Reverse order for right side
        
        for j in range(N):
            # Add coupling from nd1 term (scaled by ramp parameter r)
            if 0 <= j + nd1 < N:
                M_list[index, j, j + nd1] += mu * V1 * np.exp(-1j * b * x) * r
                M_list[index, j + nd1, j] += mu * V1 * np.exp(1j * b * x) * r
            
            # Add coupling from nd2 term (scaled by ramp parameter r)
            if 0 <= j + nd2 < N:
                M_list[index, j, j + nd2] += mu * V2 * np.exp(-1j * b * x) * r
                M_list[index, j + nd2, j] += mu * V2 * np.exp(1j * b * x) * r
            
            # Add diagonal term with ramped energy shift
            M_list[index, j, j] += 2 * mu * (-E - (j - frqcutoff_oneside) * omega + shift * r)
    
    return [x_mesh, dx, N, M_list]