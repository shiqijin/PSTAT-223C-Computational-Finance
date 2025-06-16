import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')


def black_scholes_call(S0, K, r, sigma, T, d=0):
    """Black-Scholes call option price"""
    if T <= 0:
        return max(S0 - K, 0)

    d1 = (np.log(S0 / K) + (r - d + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S0 * np.exp(-d * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return max(call_price, 0)


def explicit_fd_2d_basket_option(S1_0=40, S2_0=40, K=40, T=1.0, r=0.06,
                                 d1=0, d2=0.03, sigma1=0.3, sigma2=0.2, rho=0.6,
                                 S1_max=120, S2_max=120, N1=80, N2=80, M=365 * 24,
                                 option_type='european'):
    """
    Explicit FD solver for 2D Basket Call option
    option_type: 'european', 'american', or exercise frequency (float)
    """
    # Grid setup
    dS1 = S1_max / N1
    dS2 = S2_max / N2
    dt = T / M

    # Determine exercise schedule
    if option_type == 'european':
        exercise_times = set()  # No early exercise
    elif option_type == 'american':
        exercise_times = set(range(M))  # Exercise at every time step
    else:
        # Bermudan with specific frequency
        exercise_freq = float(option_type)
        exercise_step_freq = max(1, int(round(exercise_freq / dt)))
        exercise_times = set(range(exercise_step_freq, M, exercise_step_freq))

    S1_grid = np.linspace(0, S1_max, N1 + 1)
    S2_grid = np.linspace(0, S2_max, N2 + 1)

    # Initialize option value matrix
    V = np.zeros((M + 1, N1 + 1, N2 + 1))

    # Terminal condition: Basket call payoff at maturity
    for i in range(N1 + 1):
        for j in range(N2 + 1):
            basket_price = (S1_grid[i] + S2_grid[j]) / 2
            V[M, i, j] = max(basket_price - K, 0)

    def apply_boundary_conditions(V_current, time_step):
        """Apply boundary conditions"""
        remaining_time = (M - time_step) * dt
        present_time = time_step * dt

        # When S1 = 0: option becomes call on S2/2
        for j in range(N2 + 1):
            if remaining_time > 0:
                V_current[0, j] = black_scholes_call(S2_grid[j] / 2,
                                                     K, r, sigma2, remaining_time, d2)
            else:
                V_current[0, j] = max(S2_grid[j] / 2 - K, 0)

        # When S2 = 0: option becomes call on S1/2
        for i in range(N1 + 1):
            if remaining_time > 0:
                V_current[i, 0] = black_scholes_call(S1_grid[i] / 2,
                                                     K, r, sigma1, remaining_time, d1)
            else:
                V_current[i, 0] = max(S1_grid[i] / 2 - K, 0)

        # When S1 = S1_max or S2 = S2_max: deep in the money
        for j in range(N2 + 1):
            basket_price = (S1_max + S2_grid[j]) / 2
            #V_current[N1, j] = max(basket_price - K*np.exp(-r*remaining_time), basket_price - K)
            #V_current[N1, j] = (basket_price - K)
            V_current[N1, j] = basket_price - K * np.exp(-r * remaining_time)

        for i in range(N1 + 1):
            basket_price = (S1_grid[i] + S2_max) / 2
            #V_current[i, N2] = max(basket_price - K*np.exp(-r*remaining_time), basket_price - K)
            #V_current[i, N2] = (basket_price - K)
            V_current[N1, j] = basket_price - K * np.exp(-r * remaining_time)

    # Backward time stepping
    for m in range(M - 1, -1, -1):
        V_new = np.zeros((N1 + 1, N2 + 1))

        # Apply explicit FD scheme for interior points
        for i in range(1, N1):
            for j in range(1, N2):
                # Get values from previous time step
                V_center = V[m + 1, i, j]
                V_ip1_j = V[m + 1, i + 1, j]
                V_im1_j = V[m + 1, i - 1, j]
                V_i_jp1 = V[m + 1, i, j + 1]
                V_i_jm1 = V[m + 1, i, j - 1]
                V_ip1_jp1 = V[m + 1, i + 1, j + 1]
                V_im1_jm1 = V[m + 1, i - 1, j - 1]
                V_ip1_jm1 = V[m + 1, i + 1, j - 1]
                V_im1_jp1 = V[m + 1, i - 1, j + 1]

                # Coefficients from the explicit FD scheme
                a_ij = 1 - dt * (r + i * i * sigma1 ** 2 + j * j * sigma2 ** 2)
                b_ij = dt / 2 * (i * (r - d1) + i * i * sigma1 ** 2)
                c_ij = dt / 2 * (-i * (r - d1) + i * i * sigma1 ** 2)
                d_ij = dt / 2 * (j * (r - d2) + j * j * sigma2 ** 2)
                e_ij = dt / 2 * (-j * (r - d2) + j * j * sigma2 ** 2)
                f_ij = rho * sigma1 * sigma2 * i * j * dt / 4

                # Explicit finite difference update
                V_new[i, j] = (a_ij * V_center +
                               b_ij * V_ip1_j +
                               c_ij * V_im1_j +
                               d_ij * V_i_jp1 +
                               e_ij * V_i_jm1 +
                               f_ij * (V_ip1_jp1 + V_im1_jm1 - V_ip1_jm1 - V_im1_jp1))

        # Apply boundary conditions BEFORE early exercise check
        apply_boundary_conditions(V_new, m)

        # Apply early exercise condition if this is an exercise date
        if m in exercise_times:
            present_time = m * dt
            for i in range(N1 + 1):
                for j in range(N2 + 1):
                    basket_price = (S1_grid[i] + S2_grid[j]) / 2
                    immediate_exercise = max(basket_price - K, 0)
                    V_new[i, j] = max(V_new[i, j], immediate_exercise)

        V[m] = V_new

    return V, S1_grid, S2_grid


# Test with your parameters
print("=== Fixed Implementation Test ===")

# Parameters
S1_0, S2_0 = 40, 40
sigma1, sigma2 = 0.3, 0.2
r, d1, d2 = 0.06, 0, 0.03
rho = 0.6
T = 1.0
K = 40

# Use smaller grid for better stability
N1, N2, M = 30, 30, 20000

# Find grid indices for initial condition
S1_grid_test = np.linspace(0, 120, N1 + 1)
S2_grid_test = np.linspace(0, 120, N2 + 1)
i_0 = np.argmin(np.abs(S1_grid_test - S1_0))
j_0 = np.argmin(np.abs(S2_grid_test - S2_0))

print(f"Grid point closest to ({S1_0}, {S2_0}): ({S1_grid_test[i_0]:.2f}, {S2_grid_test[j_0]:.2f})")

# Solve European option
print("\nSolving European option...")
V_european, _, _ = explicit_fd_2d_basket_option(
    S1_0=S1_0, S2_0=S2_0, K=K, T=T, r=r, d1=d1, d2=d2,
    sigma1=sigma1, sigma2=sigma2, rho=rho,
    N1=N1, N2=N2, M=M, option_type='european'
)
european_value = V_european[0, i_0, j_0]
print(f"European value: {european_value:.8f}")

# Solve American option
print("\nSolving American option...")
V_american, _, _ = explicit_fd_2d_basket_option(
    S1_0=S1_0, S2_0=S2_0, K=K, T=T, r=r, d1=d1, d2=d2,
    sigma1=sigma1, sigma2=sigma2, rho=rho,
    N1=N1, N2=N2, M=M, option_type='american'
)
american_value = V_american[0, i_0, j_0]
print(f"American value: {american_value:.8f}")

# Solve Bermudan options
frequencies = [1 / 6, 1 / 12, 1 / 24, 1 / 365]
freq_names = ["Every 2 months", "Monthly", "Bi-weekly", "Daily"]

print(f"\nBermudan option values:")
bermudan_values = []
for freq, name in zip(frequencies, freq_names):
    V_bermudan, _, _ = explicit_fd_2d_basket_option(
        S1_0=S1_0, S2_0=S2_0, K=K, T=T, r=r, d1=d1, d2=d2,
        sigma1=sigma1, sigma2=sigma2, rho=rho,
        N1=N1, N2=N2, M=M, option_type=freq
    )
    bermudan_value = V_bermudan[0, i_0, j_0]
    bermudan_values.append(bermudan_value)
    print(f"{name:15s}: {bermudan_value:.8f}")

print(f"\nComparison:")
print(f"European:        {european_value:.8f}")
for name, value in zip(freq_names, bermudan_values):
    print(f"{name:15s}: {value:.8f}")
print(f"American:        {american_value:.8f}")

print(f"\nPremiums over European:")
print(f"American premium: {american_value - european_value:.8f}")
for name, value in zip(freq_names, bermudan_values):
    print(f"{name:15s}: {value - european_value:.8f}")

# ========== Plotting Part ==========

print("\n=== Creating 3D Surface Plot and Stopping Region Analysis ===")

# Create meshgrid for plotting
S1_mesh, S2_mesh = np.meshgrid(S1_grid_test, S2_grid_test, indexing='ij')

# 1. 3D Surface Plot of Initial Option Value V(0, S1, S2)
fig = plt.figure(figsize=(14, 10))

# Plot American option surface
ax1 = fig.add_subplot(221, projection='3d')
surf1 = ax1.plot_surface(S1_mesh, S2_mesh, V_american[0],
                         cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
ax1.set_xlabel('S1')
ax1.set_ylabel('S2')
ax1.set_zlabel('Option Value')
ax1.set_title('American Basket Call Option Value V(0, S1, S2)')
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# Plot European option surface for comparison
ax2 = fig.add_subplot(222, projection='3d')
surf2 = ax2.plot_surface(S1_mesh, S2_mesh, V_european[0],
                         cmap='plasma', alpha=0.8, linewidth=0, antialiased=True)
ax2.set_xlabel('S1')
ax2.set_ylabel('S2')
ax2.set_zlabel('Option Value')
ax2.set_title('European Basket Call Option Value V(0, S1, S2)')
fig.colorbar(surf2, ax=ax2, shrink=0.5)

# 2. Stopping Region Analysis at t = 0.5
t_half_step = M // 2  # Time step corresponding to t = 0.5

# Calculate immediate exercise value at t = 0.5
present_time_half = 0.5
immediate_exercise_half = np.zeros((N1 + 1, N2 + 1))
for i in range(N1 + 1):
    for j in range(N2 + 1):
        basket_price = (S1_grid_test[i] + S2_grid_test[j]) / 2
        immediate_exercise_half[i, j] = max(basket_price - K, 0)

# Determine stopping region (where American value equals immediate exercise value)
continuation_value = V_american[t_half_step]
stopping_region = np.abs(continuation_value - immediate_exercise_half) < 1e-9

# Plot stopping region
ax3 = fig.add_subplot(223)
contour = ax3.contourf(S1_mesh, S2_mesh, stopping_region.astype(int),
                       levels=[0, 0.5, 1], colors=['lightblue', 'red'], alpha=0.7)
ax3.contour(S1_mesh, S2_mesh, stopping_region.astype(int),
            levels=[0.5], colors=['black'], linewidths=2)
ax3.set_xlabel('S1')
ax3.set_ylabel('S2')
ax3.set_title('Stopping Region at t = 0.5\n(Red = Exercise, Blue = Continue)')
ax3.grid(True, alpha=0.3)

# Add current spot price marker
ax3.plot(S1_0, S2_0, 'ko', markersize=8, label=f'Initial: ({S1_0}, {S2_0})')
ax3.legend()

# 3. Exercise boundary visualization
ax4 = fig.add_subplot(224)
# Plot the difference between American and European values
early_exercise_premium = V_american[0] - V_european[0]
contour_levels = np.linspace(0, np.max(early_exercise_premium), 10)
cs = ax4.contourf(S1_mesh, S2_mesh, early_exercise_premium,
                  levels=contour_levels, cmap='Reds')
ax4.contour(S1_mesh, S2_mesh, early_exercise_premium,
            levels=contour_levels, colors='black', linewidths=0.5, alpha=0.5)
fig.colorbar(cs, ax=ax4)
ax4.set_xlabel('S1')
ax4.set_ylabel('S2')
ax4.set_title('Early Exercise Premium\n(American - European)')
ax4.plot(S1_0, S2_0, 'ko', markersize=8, label=f'Initial: ({S1_0}, {S2_0})')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# ========== Strict Stopping Region Plot ==========
strict_stopping = (immediate_exercise_half > 0) & (continuation_value <= immediate_exercise_half)

plt.figure(figsize=(6, 5))
cs = plt.contourf(
    S1_mesh, S2_mesh, strict_stopping,
    levels=[-0.5, 0.5, 1.5],
    colors=['lightblue', 'red'],
    alpha=0.7
)
# draw the boundary
plt.contour(
    S1_mesh, S2_mesh, strict_stopping,
    levels=[0.5],
    colors='black',
    linewidths=2
)
plt.xlabel('S1')
plt.ylabel('S2')
plt.title('Strict Stopping Region at t = 0.5\n(Immediate Exercise > 0 & V_cont ≤ Payoff)')

# now pass the QuadContourSet `cs` into colorbar
cbar = plt.colorbar(cs, ticks=[0, 1], label='0=Hold, 1=Exercise')

plt.plot(S1_0, S2_0, 'ko', markersize=8,
         label=f'Initial: ({S1_0}, {S2_0})')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
plt.show()

# Test with additional parameters
print("=== Test with small interest rate but large dividend rate ===")

# Parameters
r, d1, d2 = 0.02, 0.04, 0.1

# Use smaller grid for better stability
N1, N2, M = 80, 80, 2000

print(f"Grid point closest to ({S1_0}, {S2_0}): ({S1_grid_test[i_0]:.2f}, {S2_grid_test[j_0]:.2f})")

# Solve European option
print("\nSolving European option...")
V_european, _, _ = explicit_fd_2d_basket_option(
    S1_0=S1_0, S2_0=S2_0, K=K, T=T, r=r, d1=d1, d2=d2,
    sigma1=sigma1, sigma2=sigma2, rho=rho,
    N1=N1, N2=N2, M=M, option_type='european'
)
european_value = V_european[0, i_0, j_0]
print(f"European value: {european_value:.8f}")

# Solve American option
print("\nSolving American option...")
V_american, _, _ = explicit_fd_2d_basket_option(
    S1_0=S1_0, S2_0=S2_0, K=K, T=T, r=r, d1=d1, d2=d2,
    sigma1=sigma1, sigma2=sigma2, rho=rho,
    N1=N1, N2=N2, M=M, option_type='american'
)
american_value = V_american[0, i_0, j_0]
print(f"American value: {american_value:.8f}")

# Solve Bermudan options
print(f"\nBermudan option values:")
bermudan_values = []
for freq, name in zip(frequencies, freq_names):
    V_bermudan, _, _ = explicit_fd_2d_basket_option(
        S1_0=S1_0, S2_0=S2_0, K=K, T=T, r=r, d1=d1, d2=d2,
        sigma1=sigma1, sigma2=sigma2, rho=rho,
        N1=N1, N2=N2, M=M, option_type=freq
    )
    bermudan_value = V_bermudan[0, i_0, j_0]
    bermudan_values.append(bermudan_value)
    print(f"{name:15s}: {bermudan_value:.8f}")

print(f"\nComparison:")
print(f"European:        {european_value:.8f}")
for name, value in zip(freq_names, bermudan_values):
    print(f"{name:15s}: {value:.8f}")
print(f"American:        {american_value:.8f}")

print(f"\nPremiums over European:")
print(f"American premium: {american_value - european_value:.8f}")
for name, value in zip(freq_names, bermudan_values):
    print(f"{name:15s}: {value - european_value:.8f}")