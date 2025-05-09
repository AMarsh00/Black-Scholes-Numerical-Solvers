import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 0.05  # risk-free rate
sigma = 0.2  # volatility
alpha = 1.5  # fractional derivative exponent
S_min = 0.1  # minimum stock price
S_max = 200  # maximum stock price
T = 1.0  # time to maturity
N_x = 512  # number of spatial grid points
N_t = 100  # number of time steps
K = 100  # strike price for the option

# Compute grid spacing and time step
dx = (S_max - S_min) / (N_x - 1)
dt = T / N_t

# Spatial grid
x_grid = np.linspace(np.log(S_min), np.log(S_max), N_x)

# Initial condition (European call option payoff)
def psi(S):
    return np.maximum(S - K, 0)

# Initialize A with the payoff at time t=0
A = np.zeros(N_x)
A[:] = psi(np.exp(x_grid))

# Fourier transform for fractional derivative
def fractional_derivative(A, alpha, dx):
    k = np.fft.fftfreq(N_x, dx) * 2 * np.pi
    k = np.fft.fftshift(k)  # Centering zero frequency
    i_k_alpha = np.sign(k) * np.abs(k)**alpha
    A_hat = np.fft.fft(A)
    A_hat = A_hat * np.exp(-0.5 * sigma**alpha * i_k_alpha * dt)
    return np.fft.ifft(A_hat)

# Lie splitting method with fractional derivative and transport
def lie_splitting_method(A, r, sigma, alpha, dx, dt, N_t):
    results = np.zeros((N_t, N_x))
    for n in range(N_t):
        # Step 1: Fractional diffusion step
        A_new = fractional_derivative(A, alpha, dx)
        
        # Step 2: Transport and growth step (Euler method)
        A_new *= np.exp(r * dt)  # Forward step for the transport term
        
        # Boundary conditions
        A_new[0] = 0  # Boundary at S=0
        A_new[-1] = np.exp(x_grid[-1]) - K * np.exp(-r * (T - n * dt))  # Asymptotic behavior for large S
        
        # Update solution
        A[:] = A_new
        results[n, :] = A

    return results

# Solve the PDE using the Lie splitting method
A_results = lie_splitting_method(A, r, sigma, alpha, dx, dt, N_t)

# Plot the results at each time step in a 3D surface plot
S_vals = np.exp(x_grid)  # Stock price values

# Create meshgrid for plotting
T_vals = np.linspace(0, T, N_t)  # Time values
S_grid, T_grid = np.meshgrid(S_vals, T_vals)

# Plot the surface
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_grid, T_grid, A_results, cmap='viridis')

# Set labels and title
ax.set_xlabel('Stock Price (S)')
ax.set_ylabel('Time')
ax.set_zlabel('Option Price')
ax.set_title('Option Price Evolution over Time (Lie Splitting Scheme)')

plt.tight_layout()
plt.show()
