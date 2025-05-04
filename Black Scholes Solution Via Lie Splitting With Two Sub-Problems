import numpy as np
from scipy.fft import fft, ifft
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import matplotlib.pyplot as plt

def lie_splitting_method(T, N_t, N, r, sigma, alpha, S_max, S_min, dt, dx, psi):
    """
    Solves the option pricing problem using Lie splitting, Fourier transforms, and the method of characteristics.

    Parameters:
    - T: Total time (final time).
    - N_t: Number of time steps.
    - N: Number of spatial points.
    - r: Risk-free rate.
    - sigma: Volatility.
    - alpha: Fractional derivative exponent.
    - S_max: Maximum value of the stock price.
    - S_min: Minimum value of the stock price.
    - dt: Time step.
    - dx: Spatial step.
    - psi: Initial condition (initial option prices at time t=0).

    Returns:
    - A: The solution at each time step.
    """
    # Discretized spatial grid and corresponding wave numbers (Fourier space)
    S = np.linspace(S_min, S_max, N)
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # Wave numbers for FFT
    k = np.fft.fftshift(k)  # Centering the zero frequency

    # Handling fractional power for negative k using absolute values and signs
    ik_alpha = 1j * np.sign(k) * np.abs(k)**alpha  # Pre-compute (ik)^alpha

    # Initial condition: Apply FFT to the initial condition
    A = np.zeros((N_t, N), dtype=complex)
    A[0, :] = psi(S)  # Initial condition (at time t=0)
    
    """# Debugging step: Plot the initial condition
    plt.plot(S, A[0, :].real, label="Initial Condition (Real Part)")
    plt.xlabel("Stock Price (S)")
    plt.ylabel("Option Price")
    plt.title("Initial Condition")
    plt.show()"""

    # Time-stepping loop
    for n in range(1, N_t):
        # Step 1: Solve for u at the next time step using O_2 (Fourier transform evolution)
        A_hat = fft(A[n-1, :])  # Take FFT of previous solution
        A_hat = A_hat * np.exp(-0.5 * sigma**alpha * (ik_alpha) * dt)  # Apply operator O_2 (evolution in Fourier space)
        u_next = ifft(A_hat)  # Apply inverse FFT to get u at the next time step

        # Step 2: Solve for A using O_1 (Method of characteristics)
        # Update A using the characteristics method (no interpolation needed)
        A[n, :] = np.exp(r * dt) * u_next  # Apply the method of characteristics to update A

        """# Debugging: Check if A[n, :] is diverging to zero
        if np.all(np.real(A[n, :]) < 1e-10):  # If all values are near zero
            print(f"Warning: Solution near zero at timestep {n}.")
            break  # Exit if solution is too small"""

    return A


# Parameters for the problem
T = 1.0  # Total time
N_t = 100  # Number of time steps
N = 128  # Number of spatial grid points
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility
alpha = 1.5  # Fractional derivative exponent
S_max = 200  # Maximum price of the stock
S_min = 0  # Minimum price of the stock
dt = T / N_t  # Time step
dx = (S_max - S_min) / (N - 1)  # Spatial step
K = 100  # Strike price for the option

# Initial condition: Option payoff at time t=0 (e.g., European call option)
def psi(S):
    return np.maximum(S - K, 0)  # European call option payoff

# Run the solver
A = lie_splitting_method(T, N_t, N, r, sigma, alpha, S_max, S_min, dt, dx, psi)

# Post-processing: Plotting the results in 3D for different time steps
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Time steps to visualize
time_steps = [0, int(N_t/4), int(N_t/2), int(3*N_t/4), N_t-1]

# Create meshgrid for plotting
S_vals = np.linspace(S_min, S_max, N)  # Stock price values (x-axis)
T_vals = np.array(time_steps)  # Time step indices (y-axis)
S_grid, T_grid = np.meshgrid(S_vals, T_vals)  # Meshgrid for stock price and time

# Z values: Option prices at each time step
Z = np.real(A[time_steps, :])  # Real part of the option prices for the chosen time steps

# Plot the surface
ax.plot_surface(S_grid, T_grid, Z, cmap='viridis')

# Set labels and title
ax.set_xlabel('Stock Price (S)')
ax.set_ylabel('Time Step')
ax.set_zlabel('Option Price')
ax.set_title('Option Price Evolution over Time')

plt.tight_layout()
plt.show()
