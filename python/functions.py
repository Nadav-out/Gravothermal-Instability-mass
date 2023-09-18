# Import necessary libraries and modules
import sys
import numpy as np
from scipy.interpolate import interp1d

# Append parent directory to system path
sys.path.append("..")

# Load data from CSV files
x_lis = np.loadtxt('./data/v_over_w_lis.csv')     # List of v/w ratios
K_eff_2_lis = np.loadtxt('./data/K_5_2_lis.csv')  # K_eff^(2) list for SMFP
K_5_lis = np.loadtxt('./data/K_5_lis.csv')        # K_5 list for LMFP

# Calculate log slopes for SMFP and LMFP
n_eff_2_lis = -np.gradient(np.log10(K_eff_2_lis), np.log10(x_lis))
n_5_lis = -np.gradient(np.log10(K_5_lis), np.log10(x_lis))

# Interpolations using cubic splines
K_eff_2_int = interp1d(x_lis, K_eff_2_lis, kind='cubic')
K_5_int = interp1d(x_lis, K_5_lis, kind='cubic')
n_eff_2_int = interp1d(x_lis, n_eff_2_lis, kind='cubic')
n_5_int = interp1d(x_lis, n_5_lis, kind='cubic')
x_of_n_5_int = interp1d(n_5_lis, x_lis, kind='cubic')
x_of_n_eff_2_int = interp1d(n_eff_2_lis, x_lis, kind='cubic')

# Constants
a = 4 / np.sqrt(np.pi)
b = 25 * np.sqrt(np.pi) / 32
G_N_1 = 9.86174e13  # Newton's constant in certain units
G_N_2 = 4.30088e-6  # Newton's constant in certain units
c_light = 299792    # Speed of light in km/s

# Reference values
C_ref = 0.6
alpha_ref = 2.2
xi_ref = 0.11

# LMFP conversion factors
rho_s_to_rho_c_0 = 2.4
V_max_to_v_c_0 = 0.64



# Functions
def sigma_hat(rho_s, V_max, w, sigma_over_m, C=C_ref):
    """
    Calculates the dimensionless cross-section, sigma_hat.
    
    Parameters:
        rho_s (float or np.array): Scalar amplitude of the dark matter halo profile.
        V_max (float or np.array): Maximum circular velocity.
        w (float or np.array): The cross-section velocity scale.
        sigma_over_m (float or np.array): Ratio of cross-section sigma to mass m.
        C (float, optional): Constant C. Default is 0.64.
        
    Notes:
        - All input parameters can be either single float values or NumPy arrays.
        - The function will issue a warning if sigma_hat > 0.1 for any entries.
    
    Returns:
        float or np.array: The dimensionless cross-section, sigma_hat, as a float 
                           or as a NumPy array, depending on input type.
    """
    # Calculate velocity normalization
    vc0 = V_max_to_v_c_0 * V_max
    # Calculate density normalization
    rhoc0 = rho_s_to_rho_c_0 * rho_s
    # Dimensionless velocity parameter
    x = vc0 / w
    # Interpolated values for K5 and Keff_2
    K5 = K_5_int(x)
    Keff_2 = K_eff_2_int(x)
    # Prefactor for sigma_hat
    pref = a * C * rhoc0 / (4 * np.pi * G_N_1 * b)
    # Compute sigma squared
    sig_sq = pref * K5 * Keff_2 * (vc0 * sigma_over_m)**2
    # Warning if sigma_hat exceeds 0.1
    if np.max(sig_sq) > 1e-2:
        print("some entries with sigma_hat>0.1, proceed with caution")
    return np.sqrt(sig_sq)

def v_rho_n_LS(rho_s, V_max, w, sigma_over_m, C=C_ref, alpha=alpha_ref):
    """
    Calculates the velocity, density, and n_LS parameter at the transition point 
    from Short-to-Long (LS) mean free path.
    
    Parameters:
        rho_s (float or np.array): Scalar amplitude of the dark matter halo profile.
        V_max (float or np.array): Maximum circular velocity.
        w (float or np.array): The cross-section velocity scale.
        sigma_over_m (float or np.array): Ratio of cross-section sigma to mass m.
        C (float, optional): Constant C. Default is 0.64.
        alpha (float, optional): Power-law index. Default is 2.2.
        
    Notes:
        - All input parameters can be either single float values or NumPy arrays.
    
    Returns:
        tuple: A tuple containing:
            - v_c_LS (float or np.array): Velocity at the LS transition point.
            - rho_c_LS (float or np.array): Density at the LS transition point.
            - n_LS (float or np.array): The log-slope of the cross-section at the LS transition point.
    """
    # Calculate sigma_hat using previously defined function
    sig_hat = sigma_hat(rho_s, V_max, w, sigma_over_m, C)
    # Calculate velocity and density normalization
    vc0 = V_max_to_v_c_0 * V_max
    rhoc0 = rho_s_to_rho_c_0 * rho_s
    # Dimensionless velocity parameter
    x = vc0 / w
    # Calculate delta parameter
    delta = 1 - (n_5_int(x) + n_eff_2_int(x)) / 2 + alpha / (alpha - 2)
    # Calculate velocity, density and n_LS at the LS transition point
    v_c_LS = vc0 * sig_hat ** (-1 / delta)
    rho_c_LS = rhoc0 * (v_c_LS / vc0) ** (2 * alpha / (alpha - 2))
    n_LS = n_eff_2_int(v_c_LS / w)
    return v_c_LS, rho_c_LS, n_LS

def v_c_10_LS_fit(n_LS):
    """
    Fitting function for the central velocity v_c_10 at gamma = 10,
    normalized to the central velocity v_c_LS at the Long-to-Short (LS) mean free path transition.
    The model is based on empirical results from our paper.

    Parameters:
        n_LS (float or np.array): The log-slope of the cross-section at the LS transition point.

    Returns:
        float or np.array: The ratio v_c_10 / v_c_LS.

    Note:
        - The coefficients A, B, c, and d are empirically determined from our paper.
    """
    A = 0.74
    B = 0.008
    c = 2.526
    d = 0.026
    num = A + B * n_LS ** c
    den = n_LS ** d
    return np.exp(num / den)


def rho_c_10_LS_fit(n_LS):
    """
    Fitting function for the central density rho_c_10 at gamma = 10,
    normalized to the central density rho_c_LS at the Long-to-Short (LS) mean free path transition.
    The model is based on empirical results from our paper.

    Parameters:
        n_LS (float or np.array): The log-slope of the cross-section at the LS transition point.

    Returns:
        float or np.array: The ratio rho_c_10 / rho_c_LS.

    Note:
        - The coefficients A, B, c, and d are empirically determined from our paper.
    """
    A = 8.43
    B = 0.178
    c = 2.21
    d = 0.01
    num = A + B * n_LS ** c
    den = n_LS ** d
    return np.exp(num / den)

def v_rho_10(rho_s, V_max, w, sigma_over_m, C=C_ref, alpha=alpha_ref):
    """
    Approximates the central velocity and central density at gamma=0 instant 
    using the previously defined fitting functions v_c_10_LS_fit() and rho_c_10_LS_fit().

    Parameters:
        rho_s (float or np.array): Scalar amplitude of the dark matter halo profile.
        V_max (float or np.array): Maximum circular velocity.
        w (float or np.array): The cross-section velocity scale.
        sigma_over_m (float or np.array): Ratio of cross-section sigma to mass m.
        C (float, optional): Constant C. Default is 0.64.
        alpha (float, optional): Constant alpha. Default is 2.2.
        
    Returns:
        tuple: (v_c_10, rho_c_10), where
            - v_c_10 (float or np.array): Approximated central velocity at gamma=0 instant.
            - rho_c_10 (float or np.array): Approximated central density at gamma=0 instant.

    Notes:
        - All input parameters can be either single float values or NumPy arrays.
        - The function leverages the Long-to-Short (LS) transition point fitting functions.
    """

    v_c_LS, rho_c_LS, n_LS = v_rho_n_LS(rho_s, V_max, w, sigma_over_m, C, alpha)
    v_c_10 = v_c_LS * v_c_10_LS_fit(n_LS)
    rho_c_10 = rho_c_LS * rho_c_10_LS_fit(n_LS)
    return v_c_10, rho_c_10


def M_c_RI(rho_s, V_max, w, sigma_over_m, C=C_ref, alpha=alpha_ref, xi=xi_ref):
    """
    Approximates the mass at the point of relativistic instability (RI) using
    the approximated central velocity and central density at gamma=0 instant.

    Parameters:
        rho_s (float or np.array): Scalar amplitude of the dark matter halo profile.
        V_max (float or np.array): Maximum circular velocity.
        w (float or np.array): The cross-section velocity scale.
        sigma_over_m (float or np.array): Ratio of cross-section sigma to mass m.
        C (float, optional): Constant C. Default is 0.64.
        alpha (float, optional): Constant alpha. Default is 2.2.
        xi (float, optional): Constant xi. Default is 0.11.

    Returns:
        float or np.array: Approximated mass at the point of relativistic instability (M_c_RI).

    Notes:
        - All input parameters can be either single float values or NumPy arrays.
    """
    # Get the centralvelocity and central density at gamma=10 instant
    v_c_10, rho_c_10 = v_rho_10(rho_s, V_max, w, sigma_over_m, C=C_ref, alpha=alpha_ref)
    pref = np.sqrt(6/np.pi) * (1 + xi)**(3/2) * G_N_2**(-3/2)
    # Mass at gamma=10
    M_c_10 = pref * v_c_10**3 / rho_c_10**0.5
    # Mass at v_c=c/3
    M_c_RI = M_c_10 * 9 * (v_c_10 / c_light)**2
    return M_c_RI

