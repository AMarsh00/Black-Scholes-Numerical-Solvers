import os
os.environ["OPENBLAS_NUM_THREADS"] = "1" # keep the notebook light
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from math import log, exp, pi

# ---------- characteristic function of log‑price --------------------------
def char_func(u, S0, r, sigma, alpha, T):
    """
    ϕ(u) = E[e^{iu log S_T}] for a symmetric α‑stable process.
    c = ½ σ^α is the scale in the Lévy exponent.
    The +c term in the drift enforces E[S_T] = S0·e^{rT}.
    """
    c = 0.5 * sigma ** alpha
    return np.exp(1j * u * (log(S0) + (r + c) * T) - c * T * np.abs(u) ** alpha)

# ---------- Carr‑Madan FFT routine ----------------------------------------
def carr_madan_fractional_call(S0, K, r, sigma, alpha, T,
    N=2**12, eta=0.25, a=1.5):
    """
    Price a European call via Carr–Madan (1999) FFT.
    Parameters
    ----------
    a : damping factor (>0), typical 1–2.
    eta : grid spacing in Fourier variable ν.
    N : number of FFT points (power of two preferred).
    """
    k = log(K) # log‑strike (NB: *not* log(K/S0))
    v = np.arange(N) * eta

    # Simpson weights for the integral
    w = np.ones(N)
    w[0] = w[-1] = 1
    w[1:-1:2] = 4
    w[2:-1:2] = 2

    u = v - 1j * (a + 1)
    numer = np.exp(-1j * v * k) * char_func(u, S0, r, sigma, alpha, T)
    denom = (a**2 + a - v**2) + 1j * (2*a + 1) * v
    integrand = numer / denom

    integral = (eta / 3.0) * np.sum(w * integrand) # Simpson rule
    return exp(-a * k) * integral.real / pi

# ---------- example parameters & price ------------------------------------
S0, K = 100.0, 100.0 # spot and strike
r, T = 0.05, 1.0 # risk‑free rate, maturity (years)
sigma = 0.20 # “volatility” scale in fractional sense
alpha = 1.50 # fractional order (1 < α < 2)

price = carr_madan_fractional_call(S0, K, r, sigma, alpha, T)
print(f"Space‑fractional BS call price (α={alpha}): {price:.4f}")

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1" # keep the notebook light
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from math import log, exp, pi

# ---------- characteristic function of log‑price --------------------------
def char_func(u, S0, r, sigma, alpha, T):
    """
    ϕ(u) = E[e^{iu log S_T}] for a symmetric α‑stable process.
    c = ½ σ^α is the scale in the Lévy exponent.
    The +c term in the drift enforces E[S_T] = S0·e^{rT}.
    """
    c = 0.5 * sigma ** alpha
    return np.exp(1j * u * (log(S0) + (r + c) * T) - c * T * np.abs(u) ** alpha)

# ---------- Carr‑Madan FFT routine ----------------------------------------
def carr_madan_fractional_call(S0, K, r, sigma, alpha, T, N=2**12, eta=0.25, a=1.5):
    """
    Price a European call via Carr–Madan (1999) FFT.
    Parameters
    ----------
    a : damping factor (>0), typical 1–2.
    eta : grid spacing in Fourier variable ν.
    N : number of FFT points (power of two preferred).
    """
    k = log(K) # log‑strike (NB: *not* log(K/S0))
    v = np.arange(N) * eta

    # Simpson weights for the integral
    w = np.ones(N)
    w[0] = w[-1] = 1
    w[1:-1:2] = 4
    w[2:-1:2] = 2

    u = v - 1j * (a + 1)
    numer = np.exp(-1j * v * k) * char_func(u, S0, r, sigma, alpha, T)
    denom = (a**2 + a - v**2) + 1j * (2*a + 1) * v
    integrand = numer / denom

    integral = (eta / 3.0) * np.sum(w * integrand) # Simpson rule
    return exp(-a * k) * integral.real / pi

# ---------- example parameters & price ------------------------------------
S0, K = 100.0, 100.0 # spot and strike
r, T = 0.05, 1.0 # risk‑free rate, maturity (years)
sigma = 0.20 # “volatility” scale in fractional sense
alpha = 1.50 # fractional order (1 < α < 2)

price = carr_madan_fractional_call(S0, K, r, sigma, alpha, T)
print(f"Space‑fractional BS call price (α={alpha}): {price:.4f}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
S0 = 100.0
r = 0.05
sigma = 0.20
alpha = 1.50

# Define ranges
T_values = np.linspace(0.01, 1.0, 50)  # Time to maturity from 0.01 to 2 years
K_values = np.linspace(1, 200, 50)    # Strike prices

# Meshgrid for surface plot
T_grid, K_grid = np.meshgrid(T_values, K_values)
prices = np.zeros_like(T_grid)

# Compute prices
for i in range(T_grid.shape[0]):
    for j in range(T_grid.shape[1]):
        prices[i, j] = carr_madan_fractional_call(S0, K_grid[i, j], r, sigma, alpha, T_grid[i, j])

# Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(K_grid, T_grid, prices, cmap='viridis')

ax.set_title(f"Fractional Black-Scholes Call Price Surface (α={alpha})")
ax.set_xlabel("Strike Price S")
ax.set_ylabel("Time to Maturity T")
ax.set_zlabel("Call Option Price")
plt.show()
