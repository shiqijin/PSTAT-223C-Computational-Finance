import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Model parameters
r, sigma, gamma = 0.05, 0.4, 0.8
T, T1, T2   = 0.5, 0.25, 0.5
L1, L2     = 15.0, 25.0
X0         = 20.0
K_call     = 20.0   # strike of underlying call
K_comp     = 2.0    # strike of compound option
dt         = 0.01
x_max, x_min=100, 0

def build_operator(x, r, sigma, gamma, dx):
    """Construct lower, main, upper diagonals for L."""
    nx = len(x)
    lower = np.zeros(nx-1)
    main  = np.zeros(nx)
    upper = np.zeros(nx-1)
    for i in range(1,nx-1):
        xi = x[i]
        a  = 0.5 * sigma**2 * xi**(2*gamma)
        b  = r * xi
        lower[i-1] =  a/dx**2 - b/(2*dx)
        main[i]    = -2*a/dx**2 - r
        upper[i]   =  a/dx**2 + b/(2*dx)
    main[0] = main[-1] = 1.0  # absorbing BC
    return lower, main, upper

def implicit_fd(x, dt, r, sigma, gamma, T_horizon, V_T, V_Xmax, V_Xmin):
    """
    Fully-implicit FD: (I - dt*L) V^{m+1} = V^m, marching backward from V_T.
    """
    nx = len(x)
    N  = int(T_horizon/dt)
    dx = x[1]-x[0]
    lower, main, upper = build_operator(x,r,sigma,gamma,dx)
    A = sp.diags([-dt*lower, 1-dt*main, -dt*upper],
                 offsets=[-1,0,1], format='csc')
    V = V_T.copy()
    V[0], V[-1] = V_Xmin[-1], V_Xmax[-1]
    for i in range(N):
        V = spla.spsolve(A, V)
        V[0], V[-1] = V_Xmin[-i-2], V_Xmax[-i-2]
    return V

def crank_nicolson(x, dt, r, sigma, gamma, T_horizon, V_T, V_Xmax, V_Xmin):
    """
    Crank-Nicolson FD: (I - dt/2 L)V^{m+1} = (I + dt/2 L)V^m.
    """
    nx = len(x)
    N  = int(T_horizon/dt)
    dx = x[1]-x[0]
    lower, main, upper = build_operator(x,r,sigma,gamma,dx)
    Lmat = sp.diags([lower, main, upper], offsets=[-1,0,1], format='csc')
    I = sp.eye(nx, format='csc')
    M1 = I - 0.5*dt*Lmat
    M2 = I + 0.5*dt*Lmat
    V = V_T.copy()
    V[0], V[-1] = 0.0, 0.0
    for i in range(N):
        b = M2.dot(V)
        b[0], b[-1] = V_Xmin[-1], V_Xmax[-1] # Updating the boundary ones
        V = spla.spsolve(M1, b)
        V[0], V[-1] = V_Xmin[-i-2], V_Xmax[-i-2] # Updating the boundary ones
    return V

# === Part 1(a)&(b): Corridor option ===
for dx in [0.1, 0.02]:
    x = np.linspace(L1, L2, int((L2-L1)/dx)+1)
    V_Xmin = np.zeros(int(T/dt)+1)
    V_Xmax = np.zeros(int(T/dt)+1)
    # (a) Implicit
    V_imp = implicit_fd(x, dt, r, sigma, gamma, T, np.ones_like(x), V_Xmax, V_Xmin)
    v_imp = np.interp(X0, x, V_imp)
    # (b) Crank–Nicolson
    V_cn  = crank_nicolson(x, dt, r, sigma, gamma, T, np.ones_like(x), V_Xmax, V_Xmin)
    v_cn  = np.interp(X0, x, V_cn)
    print(f"dx={dx:.3f} → Corridor: Implicit V(0,20)={v_imp:.6f}, CN V(0,20)={v_cn:.6f}")

# === Part 1(c): Compound option ===



for dx in [0.1, 0.02]:
    x = np.linspace(x_min, x_max, int((x_max-x_min)/dx)+1)
    V_Xmin_call = np.zeros(int((T2-T1)/dt)+1)
    V_Xmax_call = np.zeros(int((T2-T1)/dt)+1)
    for i in range(int((T2-T1)/dt)+1):
        V_Xmax_call[i] = x_max -K_call * np.exp(-r * (T2- T1*(i*dt)))
    # Step 1: Underlying call from T2→T1
    call_payoff   = np.maximum(x - K_call, 0)
    # Implicit
    C_at_T1      = implicit_fd(x, dt, r, sigma, gamma, T2-T1, call_payoff, V_Xmax_call, V_Xmin_call)
    # Crank-Nicolson
    C_at_T1_CN      = crank_nicolson(x, dt, r, sigma, gamma, T2-T1, call_payoff, V_Xmax_call, V_Xmin_call)
    # Step 2: Compound payoff at T1, then back to t=0
    V_Xmin_comp = np.zeros(int((T1)/dt)+1)
    V_Xmax_comp = np.zeros(int((T1)/dt)+1)
    for i in range(int((T1)/dt)+1):
        V_Xmax_comp[i] = x_max -K_call * np.exp(-r * (T1*(i*dt)))
    # Implicit
    comp_payoff   = np.maximum(C_at_T1 - K_comp, 0)
    V_compound   = implicit_fd(x, dt, r, sigma, gamma, T1, comp_payoff, V_Xmax_comp, V_Xmin_comp)
    v_comp       = np.interp(X0, x, V_compound)
    # Crank-Nicolson
    comp_payoff_CN   = np.maximum(C_at_T1_CN- K_comp, 0)
    V_compound_CN   = crank_nicolson(x, dt, r, sigma, gamma, T1, comp_payoff_CN, V_Xmax_comp, V_Xmin_comp)
    v_comp_CN       = np.interp(X0, x, V_compound_CN)
    print(f"dx={dx:.3f} → Compound: Implicit W(0,20)={v_comp:.6f}, CN W(0,20)={v_comp_CN:.6f}")

# Plot (for dx=0.02)
dx = 0.02
x  = np.linspace(L1, L2, int((L2-L1)/dx)+1)
V_imp = implicit_fd(x, dt, r, sigma, gamma, T, np.ones_like(x), V_Xmax, V_Xmin)
V_cn  = crank_nicolson(x, dt, r, sigma, gamma, T, np.ones_like(x), V_Xmax, V_Xmin)

# Compound
Comp_dx = 0.02
Comp_x  = np.linspace(x_min, x_max, int((x_max-x_min)/dx)+1)
call_payoff = np.maximum(Comp_x - K_call, 0)
C_at_T1     = implicit_fd(Comp_x, dt, r, sigma, gamma, T2-T1, call_payoff, V_Xmax_call, V_Xmin_call)
comp_payoff = np.maximum(C_at_T1 - K_comp, 0)
V_compound  = implicit_fd(Comp_x, dt, r, sigma, gamma, T1, comp_payoff, V_Xmax_comp, V_Xmin_comp)

C_at_T1_CN     = implicit_fd(Comp_x, dt, r, sigma, gamma, T2-T1, call_payoff, V_Xmax_call, V_Xmin_call)
comp_payoff_CN = np.maximum(C_at_T1_CN - K_comp, 0)
V_compound_CN  = implicit_fd(Comp_x, dt, r, sigma, gamma, T1, comp_payoff_CN, V_Xmax_comp, V_Xmin_comp)


plt.figure(figsize=(8,5))
plt.plot(x, V_imp,   label='Implicit (corridor)')
plt.axvline(X0, color='k', ls='--', label='X0=20')
plt.xlabel('Asset price $x$')
plt.ylabel('$V(0,x)$')
plt.title('Corridor option price with Implicit Method Delta x = 0.02')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(x, V_cn,    label='Crank–Nicolson (corridor)')
plt.axvline(X0, color='k', ls='--', label='X0=20')
plt.xlabel('Asset price $x$')
plt.ylabel('$V(0,x)$')
plt.title('Corridor option price with CN Method Delta x = 0.02')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(Comp_x[500:2000], V_compound[500:2000], label='Compound option: Implicit Method')
plt.plot(Comp_x[500:2000], V_compound_CN[500:2000], label='Compound option: CN')
plt.axvline(X0, color='k', ls='--', label='X0=20')
plt.xlabel('Asset price $x$')
plt.ylabel('$V(0,x)$')
plt.title('Compound option price Delta x = 0.02')
plt.legend()
plt.grid(True)
plt.show()
