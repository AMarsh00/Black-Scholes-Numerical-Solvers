import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

def carr_madan_fft(S0, K_vals, T, t, r, sigma, alpha=1.5, a=1.5, N=2**10):
    # Step 1: Setup parameters
    tau = T - t
    c = sigma**alpha / 2
    mu = np.log(S0) + (r + c) * tau

    # Step 2: Setup frequency grid
    lambda_ = 10 * np.sqrt(c * tau)
    eta = lambda_ / N
    U_max = N * eta
    u_vals = np.arange(N) * eta

    # Step 3: Compute characteristic function phi(u)
    # For t > 0, use the characteristic function of the process
    if t > 0:
        phi_vals = np.exp(1j * u_vals * mu - c * tau * np.abs(u_vals)**alpha)
    else:
        # For t = 0, we need the characteristic function of the payoff max(S-K, 0)
        # This is a well-known form, for a call option with strike K=100
        phi_vals = np.exp(-1j * u_vals * np.log(K_vals[0])) / (1j * u_vals + a)  # Modify for the payoff here

    # Step 4: Compute g(u) for each strike K in K_vals
    g_vals_list = []
    for K in K_vals:
        g_vals = np.exp(-1j * u_vals * np.log(K)) * phi_vals / (a**2 + a - u_vals**2 + 1j * (2 * a + 1) * u_vals)
        g_vals_list.append(g_vals)

    # Step 5: Apply Simpson's weights
    weights = np.ones(N) * eta / 3
    weights[0] = eta / 2  # First value
    weights[1::2] *= 4  # Odd indices
    weights[2::2] *= 2  # Even indices

    # Step 6: Apply Simpson's weights to g(u)
    g_weighted_list = [g_vals * weights for g_vals in g_vals_list]

    # Step 7: Compute FFT of weighted g(u) for each strike
    fft_vals_list = [fft(g_weighted) for g_weighted in g_weighted_list]

    # Step 8: Compute the log-strike grid
    x_vals = -np.pi / eta + (2 * np.pi * np.arange(N)) / (N * eta)
    x_vals = np.clip(x_vals, None, 100)
    
    # Step 9: Compute strikes
    K_fft = np.exp(x_vals)

    # Step 10: Compute call prices for each strike K
    C_fft_list = []
    for fft_vals in fft_vals_list:
        C_fft = np.exp(-r * tau - a * x_vals) / np.pi * np.real(fft_vals)
        C_fft_list.append(C_fft)

    # Step 11: Directly match strikes and call prices (no interpolation)
    C_prices_list = []
    for C_fft in C_fft_list:
        # Directly match the FFT results with the desired strikes
        C_prices = np.interp(K_vals, K_fft, C_fft)
        C_prices_list.append(C_prices)
    
    return C_prices_list

# Example Usage:
S0 = 100  # Spot price
K_vals = np.linspace(100, 200, 21)  # Strike prices
T = 1  # Time to maturity
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility

# Generate spot prices (S values)
S_vals = np.linspace(1, 200, 50)  # Range of spot prices

# Initialize a 3D array to store the option prices at each S and time step (t)
time_steps = 50
C_vals_over_time = np.zeros((time_steps, len(S_vals), len(K_vals)))

# Time step increment
time_increment = T / (time_steps - 1)

# Calculate option prices for each combination of S and K at each time step
for t_index in range(time_steps):
    t = time_increment * t_index  # Linearly distribute time steps between 0 and T
    for i, S in enumerate(S_vals):
        C_prices = carr_madan_fft(S, K_vals, T, t, r, sigma)
        
        # Store prices for each strike
        for j, C in enumerate(C_prices):
            C_vals_over_time[t_index, i, j] = C[j]  # Store the price for each (S, K) pair

# Create a 3D plot for time evolution
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Prepare meshgrid for plotting
S_grid, time_grid = np.meshgrid(S_vals, np.linspace(0, T, time_steps))  # Time as the y-axis

# Take an average over all strikes for simplicity to reduce dimensionality
C_vals_transposed = C_vals_over_time[:, :, :].mean(axis=2).T  # Averaging over all strikes

# Plot the surface
C_vals_transposed = np.clip(C_vals_transposed, None, 10)
ax.plot_surface(S_grid, time_grid, C_vals_transposed, cmap='viridis')

# Labels and title
ax.set_xlabel('Spot Price (S)')
ax.set_ylabel('Time to Maturity (t)')
ax.set_zlabel('Call Option Price (V(S, t))')
ax.set_title('Evolution of Carr-Madan FFT Option Prices over Time')

# Show the plot
plt.show()
