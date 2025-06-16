import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings

warnings.filterwarnings('ignore')

# Model parameters
beta_0, beta_1, beta_2 = 0.08, -0.08, 0.5
lambda_1, lambda_2 = 62, 40
R1_0, R2_0 = -0.044, 0.007
K, T = 10, 0.33
TARGET_H = 0.001
N_STEPS = int(T / TARGET_H)
ACTUAL_H = T / N_STEPS
R2_MIN = 1e-8


def volatility_function(R1, R2):
    return beta_0 + beta_1 * R1 + beta_2 * np.sqrt(np.maximum(R2, R2_MIN))


def simulate_path(S0, Z=None, random_state=None):
    """Simulate path with optional fixed random sequence for common random numbers"""
    if Z is None:
        if random_state is not None:
            np.random.seed(random_state)
        Z = np.random.normal(0, 1, N_STEPS)

    sqrt_h = np.sqrt(ACTUAL_H)
    S = np.zeros(N_STEPS + 1)
    R1 = np.zeros(N_STEPS + 1)
    R2 = np.zeros(N_STEPS + 1)

    S[0], R1[0], R2[0] = S0, R1_0, R2_0

    for i in range(N_STEPS):
        sigma = volatility_function(R1[i], R2[i])
        S[i + 1] = S[i] + S[i] * sigma * sqrt_h * Z[i]
        R1[i + 1] = R1[i] + lambda_1 * (sigma * sqrt_h * Z[i] - R1[i] * ACTUAL_H)
        R2[i + 1] = max(R2[i] + lambda_2 * (sigma ** 2 - R2[i]) * ACTUAL_H, R2_MIN)

    return S, R1, R2


def price_call_mc(S0, M=1000, seed_offset=0):
    """Price call option using Monte Carlo with seed offset"""
    payoffs = [max(simulate_path(S0, random_state=seed_offset + m)[0][-1] - K, 0) for m in range(M)]
    return np.mean(payoffs)


def compute_delta_bump_revalue(S0, epsilon=0.005, M=1000, seed_offset=0):
    """Compute Delta using bump-and-revalue with common random numbers"""
    payoffs_up, payoffs_down = [], []

    for m in range(M):
        np.random.seed(seed_offset + m)
        Z = np.random.normal(0, 1, N_STEPS)

        S_up = simulate_path(S0 + epsilon, Z)[0]
        S_down = simulate_path(S0 - epsilon, Z)[0]

        payoffs_up.append(max(S_up[-1] - K, 0))
        payoffs_down.append(max(S_down[-1] - K, 0))

    return (np.mean(payoffs_up) - np.mean(payoffs_down)) / (2 * epsilon)


def gp_analytical_derivative(gp_model, X_test):
    """
    Compute analytical derivative of GP mean function for Matern 5/2 kernel
    """
    X_test = X_test.reshape(-1, 1)
    X_train = gp_model.X_train_
    alpha = gp_model.alpha_
    kernel = gp_model.kernel_
    length_scale = kernel.length_scale

    derivatives = np.zeros(len(X_test))

    for i, x_test in enumerate(X_test):
        K_grad = np.zeros(len(X_train))

        for j, x_train in enumerate(X_train):
            # Compute the derivative of Matern 5/2 kernel
            diff = x_test[0] - x_train[0]
            r = np.abs(diff) / length_scale

            if r < 1e-12:  # Avoid numerical issues at r=0
                K_grad[j] = 0
            else:
                sqrt5_r = np.sqrt(5) * r
                exp_term = np.exp(-sqrt5_r)

                # Matern 5/2 kernel value
                K_val = (1 + sqrt5_r + (5 * r ** 2) / 3) * exp_term

                # Derivative of Matern 5/2 kernel w.r.t. x_test
                # d/dx_test [(1 + sqrt(5)*r + 5*r^2/3) * exp(-sqrt(5)*r)]
                sign_diff = np.sign(diff)
                dr_dx = sign_diff / length_scale

                dK_dr = exp_term * (np.sqrt(5) + (10 * r) / 3 - (1 + sqrt5_r + (5 * r ** 2) / 3) * np.sqrt(5))
                K_grad[j] = dK_dr * dr_dx

        # Derivative of mean function
        derivatives[i] = np.dot(K_grad, alpha)

    return derivatives


# Main computation
print("Two-Factor Volatility Model - European Call Option Analysis")
print(f"Parameters: K={K}, T={T}, Steps={N_STEPS}, h={ACTUAL_H:.6f}")
print("=" * 60)

# Verification with discrete points
print("Verification: European call prices")
verification_S0 = [8, 9, 10, 11, 12]
for i, S0 in enumerate(verification_S0):
    price = price_call_mc(S0, M=2000, seed_offset=i * 10000)
    print(f"S0 = {S0}, Call Price = {price:.4f}")

# Main analysis range
S0_range = np.arange(8, 12.1, 0.1)
print(f"\nComputing prices and deltas for S0 range: {S0_range[0]:.1f} to {S0_range[-1]:.1f}")

# Compute prices and deltas
prices = []
deltas_bump = []

for i, S0 in enumerate(S0_range):
    if i % 10 == 0:
        print(f"Processing S0 = {S0:.1f}")

    seed_offset = i * 10000
    price = price_call_mc(S0, M=1000, seed_offset=seed_offset)
    delta = compute_delta_bump_revalue(S0, M=1000, seed_offset=seed_offset)

    prices.append(price)
    deltas_bump.append(delta)

prices = np.array(prices)
deltas_bump = np.array(deltas_bump)

# Gaussian Process surrogate with Matern 5/2 kernel as specified in problem
print("\nFitting Gaussian Process surrogate with Matern 5/2 kernel...")
X_train = S0_range.reshape(-1, 1)
kernel = Matern(length_scale=1.0, nu=2.5)  # nu=2.5 corresponds to Matern 5/2
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, n_restarts_optimizer=15)
gp.fit(X_train, prices)

print(f"Optimized GP kernel: {gp.kernel_}")

# Compute analytical GP derivative
deltas_gp = gp_analytical_derivative(gp, X_train)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Verification plot
axes[0, 0].plot(verification_S0,
                [price_call_mc(S0, M=2000, seed_offset=i * 10000) for i, S0 in enumerate(verification_S0)], 'bo-',
                markersize=6)
axes[0, 0].set_xlabel('Initial Stock Price S₀')
axes[0, 0].set_ylabel('European Call Price')
axes[0, 0].set_title('Model Verification (K=10)')
axes[0, 0].grid(True, alpha=0.3)

# Option prices
axes[0, 1].plot(S0_range, prices, 'ko', label='MC Prices', markersize=3)
y_pred, y_std = gp.predict(X_train, return_std=True)
axes[0, 1].plot(S0_range, y_pred, 'r-', label='GP Mean')
axes[0, 1].fill_between(S0_range, y_pred - 2 * y_std, y_pred + 2 * y_std,
                        alpha=0.3, color='red', label='95% CI')
axes[0, 1].set_xlabel('Initial Stock Price S₀')
axes[0, 1].set_ylabel('Option Price')
axes[0, 1].set_title('GP Fit to Option Prices')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Delta comparison
axes[1, 0].plot(S0_range, deltas_bump, 'b-o', label='Bump-and-Revalue', markersize=3)
axes[1, 0].plot(S0_range, deltas_gp, 'r-s', label='GP Analytical Derivative', markersize=3)
axes[1, 0].set_xlabel('Initial Stock Price S₀')
axes[1, 0].set_ylabel('Delta')
axes[1, 0].set_title('Delta Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Delta differences
axes[1, 1].plot(S0_range, deltas_bump - deltas_gp, 'g-o', markersize=3)
axes[1, 1].set_xlabel('Initial Stock Price S₀')
axes[1, 1].set_ylabel('Delta Difference (Bump - GP)')
axes[1, 1].set_title('Delta Method Comparison')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Results summary
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"S₀ range: {S0_range[0]:.1f} to {S0_range[-1]:.1f}")
print(f"Number of points: {len(S0_range)}")
print(f"MC simulations per point: 1000")
print(f"Bump size (ε): 0.005")
print(f"Optimized GP kernel: {gp.kernel_}")
print()
print("Delta Statistics:")
print(f"Bump-and-Revalue range: [{deltas_bump.min():.4f}, {deltas_bump.max():.4f}]")
print(f"GP Analytical Derivative range: [{deltas_gp.min():.4f}, {deltas_gp.max():.4f}]")
print(f"Mean absolute difference: {np.mean(np.abs(deltas_bump - deltas_gp)):.6f}")
print(f"Max absolute difference: {np.max(np.abs(deltas_bump - deltas_gp)):.6f}")
print()
print("Model Parameters:")
print(f"Time steps: {N_STEPS}, Actual h: {ACTUAL_H:.6f}")
print(f"R₂ minimum: {R2_MIN}")