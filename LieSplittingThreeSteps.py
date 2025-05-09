import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from mpl_toolkits.mplot3d import Axes3D

def lie_splitting_three_subproblems_no_interp(T, N_t, N, r, sigma, alpha, S_min, S_max, psi):
    dt = T / N_t
    S = np.linspace(S_min, S_max, N)
    dS = S[1] - S[0]

    k = np.fft.fftfreq(N, d=dS) * 2 * np.pi
    k = np.fft.fftshift(k)
    ik_alpha = np.fft.ifftshift(np.abs(k)**alpha * np.exp(1j * np.pi * alpha / 2 * np.sign(k)))

    A = np.zeros((N_t, N), dtype=complex)
    A[0, :] = psi(S)

    for n in range(N_t - 1):
        A_hat = fft(A[n, :])
        u_hat = A_hat * np.exp(-0.5 * sigma**alpha * ik_alpha * dt)
        u = ifft(u_hat)

        S_transformed = S * np.exp(-r * dt)
        q = np.zeros_like(u)

        for j, s_val in enumerate(S):
            s_target = s_val * np.exp(-r * dt)
            if S_min <= s_target <= S_max:
                j_prime = np.argmin(np.abs(S - s_target))
                q[j] = u[j_prime]
            else:
                q[j] = 0

        A[n + 1, :] = np.exp(r * dt) * q

    return S, A

# Parameters
T = 1.0
N_t = 100
N = 256
r = 0.05
sigma = 0.2
alpha = 1.5
S_min = 0
S_max = 200
K = 100

def psi(S):
    return np.maximum(S - K, 0)

# Run
S_grid, A_sol = lie_splitting_three_subproblems_no_interp(T, N_t, N, r, sigma, alpha, S_min, S_max, psi)

# Plotting
T_vals = np.linspace(0, T, N_t)
S_vals, T_mesh = np.meshgrid(S_grid, T_vals)
A_real = np.real(A_sol)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_vals, T_mesh, A_real, cmap='viridis')
ax.set_xlabel('Stock Price S')
ax.set_ylabel('Time t')
ax.set_zlabel('Option Price')
ax.set_title('Option Price Surface (Three-Step Lie Splitting)')
plt.tight_layout()
plt.show()
