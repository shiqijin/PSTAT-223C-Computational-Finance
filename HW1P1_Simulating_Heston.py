import numpy as np
import time

# Parameters
r = 0.05
kappa = 1.0
theta = 0.2
eta = 0.5
rho = -0.4
S0 = 100.0
V0 = 0.25
T = 1.0
K = 100.0
NT = 52
M = 100_000

# Confidence multipliers
z95 = 1.96
z99 = 2.5758

def simulate_heston(scheme, dt, S0 =S0, V0 = V0, T = T, K = K, NT = NT, M = M, r = r, kappa = kappa, theta = theta, eta = eta, rho = rho):
    """
    Simulate Heston under risk-neutral measure:
      dS = r S dt + S sqrt(V) dW1
      dV = kappa(theta - V) dt + eta sqrt(V) dW2
    scheme: 'EE' = Euler-Euler; 'EM' = Euler (S) + Milstein (V)
    dt: time step
    Returns arrays of discounted put and Asian payoffs.
    """
    n_steps = int(T / dt)
    steps_per_week = n_steps // NT
    
    S = np.full(M, S0)
    V = np.full(M, V0)
    sum_S_weekly = np.zeros(M)
    
    sqrt_dt = np.sqrt(dt)
    for step in range(1, n_steps + 1):
        # generate correlated increments
        Z1 = np.random.randn(M)
        Z2 = np.random.randn(M)
        dW1 = sqrt_dt * Z1
        dW2 = sqrt_dt * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)
        
        # volatility for diffusion at time n
        sqrtV = np.sqrt(np.maximum(V, 0))
        
        # Euler update for S using V_n
        S = S + r * S * dt + S * sqrtV * dW1
        
        # update V
        if scheme == 'EE':
            V = V + kappa * (theta - V) * dt + eta * sqrtV * dW2
        elif scheme == 'EM':
            V = (V
                 + kappa * (theta - V) * dt
                 + eta * sqrtV * dW2
                 + 0.5 * eta**2 * (dW2**2 - dt))
        V = np.maximum(V, 0)
        
        # accumulate weekly S
        if step % steps_per_week == 0:
            sum_S_weekly += S
    
    # payoffs
    discounted_put = np.exp(-r * T) * np.maximum(K - S, 0)
    avg_S = sum_S_weekly / NT
    discounted_asian = np.exp(-r * T) * np.maximum(S - avg_S, 0)
    
    return discounted_put, discounted_asian

# Run simulations and report
# for scheme_label, scheme_code in [('Euler-Euler', 'EE'), ('Euler-Milstein', 'EM')]:
 #   print(f"\nScheme: {scheme_label}")
 #   for r_exp in [1, 2, 3, 4]:
 #       dt = 1.0 / (52 * 2**r_exp)
 #       start = time.time()
 #       put_payoffs, asian_payoffs = simulate_heston(scheme_code, dt)
 #       elapsed = time.time() - start
 #       
 #       for opt_label, payoffs in [('Put', put_payoffs), ('Asian', asian_payoffs)]:
 #           mean = np.mean(payoffs)
 #           std = np.std(payoffs, ddof=1)
 #           se = std / np.sqrt(M)
 #           ci95 = (mean - z95 * se, mean + z95 * se)
 #           ci99 = (mean - z99 * se, mean + z99 * se)
            
  #          print(f"  r={r_exp}, {opt_label}: price={mean:.4f}, "
  #                f"95% CI=[{ci95[0]:.4f}, {ci95[1]:.4f}], "
  #                f"99% CI=[{ci99[0]:.4f}, {ci99[1]:.4f}], "
  #                f"time={elapsed:.2f}s")
