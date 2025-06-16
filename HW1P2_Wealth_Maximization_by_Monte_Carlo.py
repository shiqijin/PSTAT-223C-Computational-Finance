import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Model parameters
T = 2.0
r = 0.05
mu = 0.25
sigma = 0.25
lam = 6.0          # jump intensity
zeta = 50.0        # parameter of Exp(zeta), so mean jump ~ 1/zeta = 0.02
gamma = -1.5       # utility exponent
dt = 0.02          # regular time-step
M = 20000          # number of Monte Carlo paths

def simulate_one_path(pi, lamb = lam):
    """
    Simulate one path of the wealth process X under fixed allocation pi.
    Uses non-uniform grid: regular grid of step dt, plus exact jump times.
    """
    # 1) simulate jump times and sizes
    N_jumps = np.random.poisson(lamb * T)
    if N_jumps > 0:
        jump_times = np.sort(np.random.uniform(0, T, size=N_jumps))
        Yj = np.random.exponential(1/zeta, size=N_jumps)
    else:
        jump_times = np.array([])
        Yj = np.array([])

    # 2) construct the combined time grid
    reg_times = np.arange(0, T, dt)
    times = np.unique(np.concatenate((reg_times, jump_times, [T])))
    
    # 3) simulate X over this grid
    X = 1.0
    t_prev = 0.0
    j = 0  # index for jump_times
    for t in times[1:]:
        delta = t - t_prev
        # Brownian increment
        dW = np.sqrt(delta) * np.random.randn()
        # Euler update between jumps
        drift = (pi * mu + (1-pi) * r) * delta
        diffusion = pi * sigma * dW
        X *= (1 + drift + diffusion)
        # jump at t?
        if j < len(jump_times) and abs(t - jump_times[j]) < 1e-8:
            fj = 2 - np.exp(Yj[j])  # jump factor: pct drop
            X *= (1 + pi * (fj - 1))
            j += 1
        t_prev = t

    return X

def estimate_expected_utility(pi, lamd):
    """
    Estimate E[U(X_T)] for a given pi,
    using M Monte Carlo paths.
    """
    U = np.empty(M)
    for i in range(M):
        XT = simulate_one_path(pi, lamd)
        U[i] = XT**gamma / gamma
    return np.mean(U)

# 3) evaluate with pi = 0.5
EU_default = estimate_expected_utility(0.5, 6)
print(f"Expected utility of terminal wealth with pi=0.5: {EU_default}")

# 4) evaluate over a grid of pi values
pi_grid = np.linspace(0, 1, 101)
new_lam = (0, 6, 36)

for jump_rate in new_lam:
    EU_values = np.array([estimate_expected_utility(pi, jump_rate) for pi in pi_grid])

# 5) find empirical optimal pi
    pi_star = pi_grid[np.argmax(EU_values)]
    print(f"Empirical optimal π* ≈ {pi_star:.3f} with jump rate lambda={jump_rate}")

# 6) plot π ↦ E[U(X_T)]
    plt.figure(figsize=(8, 5))
    plt.plot(pi_grid, EU_values, marker='o')
    plt.xlabel("Allocation π")
    plt.ylabel("E[U(X_T)]")
    plt.title(f"Expected Utility vs π (with jumps) with jump factor lambda={jump_rate}")
    plt.grid(True)
    plt.show()





