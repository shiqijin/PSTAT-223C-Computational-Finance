import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from HW2_Problem2 import heston_explicit     # for FD baseline
from HW1_P1 import simulate_heston           # Monte Carlo simulator
from mpl_toolkits.mplot3d import Axes3D       # noqa: F401 for 3D plotting

# ──────────────────────────────────────────────────────────────────────────────
# 1) Model parameters & discretizations
# ──────────────────────────────────────────────────────────────────────────────
r, kappa, theta, eta, rho = 0.05, 1.0, 0.2, 0.5, -0.4
K, T = 100.0, 1.0
S_max, V_max = 300.0, 1.0

# Finite‐difference grid (for FD baseline)
dt_fd, ds, dv = 0.00005, 2.0, 0.125

# Monte Carlo settings
dt_mc, M_mc = 0.01, 1000

# ──────────────────────────────────────────────────────────────────────────────
# 2) FD solver → interpolator (for Part (c))
# ──────────────────────────────────────────────────────────────────────────────
def run_heston_fd():
    params = dict(
        r=r, kappa=kappa, theta=theta,
        eta=eta, rho=rho,
        K=K, T=T, S0=100.0, V0=0.25,
        S_max=S_max, V_max=V_max
    )
    return heston_explicit(**params, dt=dt_fd, ds=ds, dv=dv)

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

# ──────────────────────────────────────────────────────────────────────────────
# 3) Build Monte Carlo training set
# ──────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
n_train = 100
s0_train = np.concatenate((np.random.uniform(10, 200, int(n_train * 0.8)), np.random.uniform(0, 10, int(n_train * 0.1)), np.random.uniform(200, 300, int(n_train * 0.1))))
v0_train = np.random.uniform(0.05, 0.6, n_train)

X_train = np.column_stack([s0_train, v0_train])  # [S0, V0]
y_train = np.empty(n_train)

for i, (s0, v0) in enumerate(X_train):
    # call simulate_heston from HW1_P1; it accepts S0 and V0 as keyword args
    put_payoffs, _ = simulate_heston('EM', dt_mc,
                                     S0=s0, V0=v0, T=T, K=K,
                                     NT=52, M=M_mc,
                                     r=r, kappa=kappa, theta=theta,
                                     eta=eta, rho=rho)
    y_train[i] = np.mean(put_payoffs)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Build evaluation mesh & FD “truth”
# ──────────────────────────────────────────────────────────────────────────────
n_grid = 100
s0_grid = np.linspace(10, 200, n_grid)
v0_grid = np.linspace(0.05, 0.6, n_grid)
S0_mesh, V0_mesh = np.meshgrid(s0_grid, v0_grid)

X_eval = np.column_stack([S0_mesh.ravel(), V0_mesh.ravel()])  # [S0, V0]
pts_fd  = np.column_stack([X_eval[:,1], X_eval[:,0]])         # [V0, S0]

y_fd = fd_interp(pts_fd).reshape(S0_mesh.shape)

# ──────────────────────────────────────────────────────────────────────────────
# 5a) GP surrogate with fixed Squared‐Exponential kernel
# ──────────────────────────────────────────────────────────────────────────────
kernel_se = ConstantKernel(constant_value=10**2, constant_value_bounds="fixed") * RBF(length_scale=[10.0, 0.1], length_scale_bounds="fixed") + WhiteKernel(noise_level=1.0)
gp_se = GaussianProcessRegressor(kernel=kernel_se,
                                 n_restarts_optimizer=10,
                                 alpha=1e-8,
                                 random_state=42)
gp_se.fit(X_train, y_train)
y_se, std_se = gp_se.predict(X_eval, return_std=True)
y_se = y_se.reshape(S0_mesh.shape)

# ──────────────────────────────────────────────────────────────────────────────
# 5b) GP surrogate with Matern-5/2 & MLE‐optimized lengthscales
# ──────────────────────────────────────────────────────────────────────────────
kernel_mat = ConstantKernel() * Matern(length_scale=[10.0, 0.1], nu=2.5) + WhiteKernel(noise_level=1.0)
gp_mat = GaussianProcessRegressor(kernel=kernel_mat,
                                  n_restarts_optimizer=10,
                                  alpha=1e-8,
                                  random_state=42)
gp_mat.fit(X_train, y_train)
y_mat, std_mat = gp_mat.predict(X_eval, return_std=True)
y_mat = y_mat.reshape(S0_mesh.shape)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Compare surfaces visually
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 12), subplot_kw={'projection': '3d'})
axes[0,0].plot_surface(S0_mesh, V0_mesh, y_fd,   cmap='viridis'); axes[0,0].set_title('FD‐Interpolated')
axes[0,1].plot_surface(S0_mesh, V0_mesh, y_se,   cmap='plasma');  axes[0,1].set_title('SE GP')
axes[1,0].plot_surface(S0_mesh, V0_mesh, y_mat,  cmap='inferno'); axes[1,0].set_title('Matern GP')
diff_mat = np.abs(y_mat - y_fd)
axes[0,2].plot_surface(S0_mesh, V0_mesh, diff_mat, cmap='coolwarm'); axes[0,2].set_title('|Matern-FD| Error')
diff_se = np.abs(y_se - y_fd)
axes[1,1].plot_surface(S0_mesh, V0_mesh, diff_se, cmap='coolwarm'); axes[1,1].set_title('|SE–FD| Error')
diff_se_mat = np.abs(y_se - y_mat)
axes[1,2].plot_surface(S0_mesh, V0_mesh, diff_se_mat, cmap='coolwarm'); axes[1,2].set_title('|SE–Matern| Distance')
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 7) Quantitative metrics
# ──────────────────────────────────────────────────────────────────────────────
rmse_se  = np.sqrt(np.mean((y_se  - y_fd)**2))
rmse_mat = np.sqrt(np.mean((y_mat - y_fd)**2))
mae_se   = np.mean(np.abs(y_se  - y_fd))
mae_mat  = np.mean(np.abs(y_mat - y_fd))

metrics = pd.DataFrame({
    'SE GP':   {'RMSE': rmse_se,  'MAE': mae_se,  'LL': gp_se.log_marginal_likelihood()},
    'Matern GP': {'RMSE': rmse_mat,'MAE': mae_mat,'LL': gp_mat.log_marginal_likelihood()}
})
print(metrics)

# ──────────────────────────────────────────────────────────────────────────────
# 8) Single‐point check at (S0=100, V0=0.25)
# ──────────────────────────────────────────────────────────────────────────────
S0_pt, V0_pt = 100.0, 0.25
fd_pt  = fd_interp([[V0_pt, S0_pt]])[0]
se_pt  = gp_se.predict([[S0_pt, V0_pt]])[0]
mat_pt = gp_mat.predict([[S0_pt, V0_pt]])[0]

print(f"\nAt (S0={S0_pt}, V0={V0_pt}): FD={fd_pt:.4f}, SE_GP={se_pt:.4f}, Matern_GP={mat_pt:.4f}")
