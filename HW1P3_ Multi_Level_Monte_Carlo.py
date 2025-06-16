import numpy as np
import time

# =========================
# Problem 3: CEV Corridor Option
# =========================

# Model parameters
X0 = 20.0
sigma = 0.4
r = 0.04
gamma_exp = 0.8
T = 0.5
L1, L2 = 15.0, 25.0

# Part (a): Plain Euler
def simulate_euler(M, dt):
    steps = int(T / dt)
    sqrt_dt = np.sqrt(dt)
    payoffs = np.zeros(M)
    for i in range(M):
        X = X0
        alive = True
        for _ in range(steps):
            z = np.random.randn()
            X += r * X * dt + sigma * (X**gamma_exp) * sqrt_dt * z
            if X < L1 or X > L2:
                alive = False
                break
        payoffs[i] = alive
    return np.exp(-r * T) * payoffs

def price_euler(M, dt):
    start = time.time()
    payoffs = simulate_euler(M, dt)
    price = payoffs.mean()
    var = payoffs.var(ddof=1)
    se = np.sqrt(var / M)
    ci95 = (price - 1.96 * se, price + 1.96 * se)
    return price, ci95, var, time.time() - start

# Part (b): Antithetic Sampling
def simulate_antithetic(M, dt):
    assert M % 2 == 0, "M must be even for antithetic sampling"
    half = M // 2
    steps = int(T / dt)
    sqrt_dt = np.sqrt(dt)
    payoffs = np.zeros(half)
    for i in range(half):
        Z = np.random.randn(steps)
        # path 1
        X = X0
        alive1 = True
        for z in Z:
            X += r * X * dt + sigma * (X**gamma_exp) * sqrt_dt * z
            if X < L1 or X > L2:
                alive1 = False
                break
        # path 2 (antithetic)
        X = X0
        alive2 = True
        for z in Z:
            X += r * X * dt + sigma * (X**gamma_exp) * sqrt_dt * (-z)
            if X < L1 or X > L2:
                alive2 = False
                break
        payoffs[i] = 0.5 * (alive1 + alive2)
    return np.exp(-r * T) * payoffs

def price_antithetic(M, dt):
    start = time.time()
    payoffs = simulate_antithetic(M, dt)
    price = payoffs.mean()
    var = payoffs.var(ddof=1)
    se = np.sqrt(var / M)
    ci95 = (price - 1.96 * se, price + 1.96 * se)
    return price, ci95, var, time.time() - start

# Part (c): Multi-Level Monte Carlo with variance
def mlmc(Ns, base_dt):
    levels = len(Ns)
    price_sum = 0.0
    var_levels = []
    var_estimate = 0.0
    times_levels = []
    
    for ℓ in range(levels):
        dt_fine = base_dt / (2**ℓ)
        Nl = Ns[ℓ]
        start = time.time()
        if ℓ == 0:
            Y = simulate_euler(Nl, dt_fine)
        else:
            dt_coarse = base_dt / (2**(ℓ-1))
            Pf = simulate_euler(Nl, dt_fine)
            Pc = simulate_euler(Nl, dt_coarse)
            Y = Pf - Pc
        meanY = Y.mean()
        varY = Y.var(ddof=1)
        # accumulate
        price_sum += meanY
        var_levels.append(varY)
        var_estimate += varY / Nl
        times_levels.append(time.time() - start)
    
    # discount the sum
    price = np.exp(-r * T) * price_sum
    # variance of estimator
    var_price = np.exp(-2 * r * T) * var_estimate
    se_price = np.sqrt(var_price)
    ci95 = (price - 1.96 * se_price, price + 1.96 * se_price)
    
    return price, ci95, var_levels, var_price, times_levels

# =========================
# Run all parts
# =========================
M = 100_000
dt = 0.01
Ns = [50_000, 15_000, 5_000]

# (a)
price_e, ci_e, var_e, t_e = price_euler(M, dt)
print(f"Euler:        Price = {price_e:.6f}, 95% CI = [{ci_e[0]:.6f}, {ci_e[1]:.6f}], Var = {var_e:.3e}, Time = {t_e:.2f}s")

# (b)
price_a, ci_a, var_a, t_a = price_antithetic(M, dt)
print(f"Antithetic:   Price = {price_a:.6f}, 95% CI = [{ci_a[0]:.6f}, {ci_a[1]:.6f}], Var = {var_a:.3e}, Time = {t_a:.2f}s")

# (c)
price_mlmc, ci_mlmc, var_lvls, var_mlmc, times_mlmc = mlmc(Ns, dt)
print(f"MLMC:         Price = {price_mlmc:.6f}, 95% CI = [{ci_mlmc[0]:.6f}, {ci_mlmc[1]:.6f}], Var = {var_mlmc:.3e}")
for ℓ, (v, tm) in enumerate(zip(var_lvls, times_mlmc)):
    print(f"  Level {ℓ}: Var(Y) = {v:.3e}, Time = {tm:.2f}s")
