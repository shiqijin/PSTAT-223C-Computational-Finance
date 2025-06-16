import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from scipy.interpolate import RegularGridInterpolator
from HW2_Problem2 import heston_explicit
from mpl_toolkits.mplot3d import Axes3D   # for 3D plots
import pandas as pd

# ----------------------
# 1) Model & discretization
# ----------------------
r, kappa, theta, eta, rho = 0.05, 1.0, 0.2, 0.5, -0.4
K, T = 100, 1.0
S_max, V_max = 300, 1.0

# FD grid for baseline
dt_fd, ds, dv = 0.000005, 2, 0.125

# Monte Carlo settings
dt_mc, M = 0.01, 1000
sqrt_dt = np.sqrt(dt_mc)
n_steps = int(T / dt_mc)

# ----------------------
# 2) FD solver wrapper
# ----------------------
def run_heston_fd():
    params = dict(
        r=r, kappa=kappa, theta=theta,
        eta=eta, rho=rho,
        K=K, T=T, S0=100, V0=0.25,
        S_max=S_max, V_max=V_max
    )
    return heston_explicit(**params, dt=dt_fd, ds=ds, dv=dv)

# ----------------------
# 3) Monte Carlo pricer
# ----------------------
def simulate_mc_price(S0, V0, M=M, dt=dt_mc):
    prices = np.empty(M)
    for m in range(M):
        S, v = S0, V0
        for _ in range(n_steps):
            # draw two independent normals
            Z1, Z2 = np.random.randn(), np.random.randn()
            dW1 = sqrt_dt * Z1
            # correlate W2 with W1
            dW2 = rho * dW1 + sqrt_dt * np.sqrt(1 - rho**2) * Z2

            # --- Milstein for v ---
            sv = np.sqrt(max(v, 0))
            dv_drift = kappa * (theta - v) * dt
            dv_diff  = eta * sv * dW2
            # correction term f f' ( (dW)^2 - dt )
            dv_corr  = 0.5 * eta * (eta / (2 * sv)) * (dW2**2 - dt) if sv > 0 else 0.0
            v_new = v + dv_drift + dv_diff + dv_corr
            v = max(v_new, 0)

            # --- Euler for S ---
            S = S + r * S * dt + sv * S * dW1

        # discounted Put payoff
        prices[m] = np.exp(-r * T) * max(K - S, 0)

    return prices.mean()

# ----------------------
# 4) Build training set via MC
# ----------------------
np.random.seed(42)
n_train = 100
s0_train = np.random.uniform(50, 150, n_train)
v0_train = np.random.uniform(0.05, 0.6, n_train)

X_train = np.column_stack([s0_train, v0_train])                              # [S,V]
y_train = np.array([simulate_mc_price(s, v) for s, v in X_train])           # MC labels

# ----------------------
# 5) FD‐interpolator for “ground truth” surface
# ----------------------
fd_grid = run_heston_fd()
s_grid = np.arange(0, S_max + ds, ds)
v_grid = np.arange(0, V_max + dv, dv)

fd_interp = RegularGridInterpolator(
    (v_grid, s_grid),
    fd_grid.T,
    method="linear",
    bounds_error=False,
    fill_value=None
)

# ----------------------
# 6) Build evaluation mesh
# ----------------------
n_grid = 30
s0_grid = np.linspace(50, 150, n_grid)
v0_grid = np.linspace(0.05, 0.6, n_grid)
S0_mesh, V0_mesh = np.meshgrid(s0_grid, v0_grid)
X_eval = np.column_stack([S0_mesh.ravel(), V0_mesh.ravel()])  # shape (900,2)

# FD surface on mesh
pts_fd = np.column_stack([X_eval[:, 1], X_eval[:, 0]])        # [V,S]
y_fd = fd_interp(pts_fd).reshape(S0_mesh.shape)

# ----------------------
# 7a) GP w/ fixed SE kernel
# ----------------------
kernel_se = ConstantKernel(10**2) * RBF(length_scale=[10, 0.1]) + WhiteKernel(noise_level=1.0)
gp_se = GaussianProcessRegressor(
    kernel=kernel_se,
    n_restarts_optimizer=10,
    alpha=1e-8,
    random_state=42
)
gp_se.fit(X_train, y_train)
y_se, std_se = gp_se.predict(X_eval, return_std=True)
y_se = y_se.reshape(S0_mesh.shape)

# ----------------------
# 7b) GP w/ Matern-5/2 & MLE
# ----------------------
kernel_mat = ConstantKernel() * Matern(length_scale=[10, 0.1], nu=2.5) + WhiteKernel(noise_level=1.0)
gp_mat = GaussianProcessRegressor(
    kernel=kernel_mat,
    n_restarts_optimizer=10,
    alpha=1e-8,
    random_state=42
)
gp_mat.fit(X_train, y_train)
y_mat, std_mat = gp_mat.predict(X_eval, return_std=True)
y_mat = y_mat.reshape(S0_mesh.shape)

# ----------------------
# 8) Compare surfaces
# ----------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={'projection': '3d'})
axes[0,0].plot_surface(S0_mesh, V0_mesh, y_fd,   cmap='viridis');    axes[0,0].set_title('FD interp')
axes[0,1].plot_surface(S0_mesh, V0_mesh, y_se,   cmap='plasma');     axes[0,1].set_title('SE GP')
axes[1,0].plot_surface(S0_mesh, V0_mesh, y_mat,  cmap='inferno');    axes[1,0].set_title('Matern GP')
diff_se = np.abs(y_se - y_fd)
axes[1,1].plot_surface(S0_mesh, V0_mesh, diff_se,cmap='coolwarm');  axes[1,1].set_title('|SE–FD|')
plt.tight_layout()
plt.show()

# ----------------------
# 9) Quantitative metrics
# ----------------------
rmse_se   = np.sqrt(np.mean((y_se   - y_fd)**2))
rmse_mat  = np.sqrt(np.mean((y_mat  - y_fd)**2))
mae_se    = np.mean( np.abs(y_se   - y_fd))
mae_mat   = np.mean( np.abs(y_mat  - y_fd))
print(pd.DataFrame({
    'SE GP':   {'RMSE': rmse_se,  'MAE': mae_se,  'LL': gp_se.log_marginal_likelihood()},
    'Mat GP':  {'RMSE': rmse_mat, 'MAE': mae_mat, 'LL': gp_mat.log_marginal_likelihood()}
}))

# ----------------------
# (Optional) Predict at a single point
# ----------------------
S0_pt, V0_pt = 100.0, 0.25
fd_pt    = fd_interp([[V0_pt, S0_pt]])[0]
se_pt    = gp_se.predict([[S0_pt, V0_pt]])[0]
mat_pt   = gp_mat.predict([[S0_pt, V0_pt]])[0]
print(f"\nAt (S0={S0_pt}, V0={V0_pt}):  FD={fd_pt:.4f}, SE_GP={se_pt:.4f}, Mat_GP={mat_pt:.4f}")
