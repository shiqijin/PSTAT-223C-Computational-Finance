import numpy as np

# Model parameters
r, kappa, theta, eta, rho=0.05, 1.0, 0.2, 0.5, -0.4
K=100
T=1.0
S0, V0=100, 0.25
S_max, V_max=300, 2.0
dt = 0.01
ds = 0.02
dv = 0.02
M, N, J = int(T/dt), int(S_max/ds), int(V_max/dv)


# We generate boundaries condition of the PDE here
def boundaries_condition(T, S_max, V_max, K, r, ds, dt, dv):
    M, N, J = int(T/dt), int(S_max/ds), int(V_max/dv)
    # Terminal Condition
    P_T = np.zeros((N+1, J+1))
    for n in range(N+1):
        P_T[n,:] = np.maximum(K - n * ds, 0)

    # Boundaries for s
    P_Smax = np.zeros((M+1, J+1))
    P_Smin = np.zeros((M+1, J+1))
    for m in range(M+1):
        P_Smin[m,:] = K * np.exp(-r * (T - m * dt)) * np.ones(J+1)

    # Boundaries for v
    P_Vmax, P_Vmin = np.zeros((M+1, N+1)), np.zeros((M+1, N+1))
    for m in range(M+1):
        P_Vmax[m,:] = K * np.exp(-r * (T - m * dt)) * np.ones(N+1)
        P_Vmax[m, 0] = P_Smin[m, J] # For the corner cases, we use boundaries by s.
        P_Vmax[m, N] = P_Smax[m, J]
        P_Vmin[m, 0] = P_Smin[m, 0]
        P_Vmin[m, N] = P_Smax[m, 0]
        for n in range(N-1):
            P_Vmin[m, n+1] = np.maximum(K * np.exp(-r * (T - m * dt)) - (n + 1) * ds, 0)
    return P_T, P_Smax, P_Smin, P_Vmax, P_Vmin

# A few more constants we need in the diecretisation
def parameters(eta, rho, dt, dv, ds):
    A1 = (rho * eta * dt)/(4 * ds * dv)
    A2 = dt/(2 * ds**2)
    A3 = dt/(2 * ds)
    A4 = (eta**2 * dt)/(2 * dv**2)
    return A1, A2, A3, A4



def heston_explicit(r, kappa, theta, eta, rho, S_max, V_max, K, T, S0, V0, dt, ds, dv):
    M, N, J = int(T/dt), int(S_max/ds), int(V_max/dv)
    P_T, P_Smax, P_Smin, P_Vmax, P_Vmin = boundaries_condition(T, S_max, V_max,K, r, ds, dt, dv)

    # A few more constants we need in the diecretisation
    A1, A2, A3, A4 = parameters(eta, rho, dt, dv, ds)

    P_0 = np.zeros((N+1, J+1))
    P_0[0, :] = P_Smin[0, :]
    P_0[N, :] = P_Smax[0, :]
    P_0[:, 0] = P_Vmin[0, :]
    P_0[:, J] = P_Vmax[0, :]

    P = P_T
    for m in range(M):
        P_temp = np.zeros((N+1, J+1))
        P_temp[0, :] = P_Smin[m, :]
        P_temp[N, :] = P_Smax[m, :]
        P_temp[:, 0] = P_Vmin[m, :]
        P_temp[:, J] = P_Vmax[m, :]
        for n in range(N-1):
            n = n+1
            s = n * ds
            for j in range(J-1):
                j = j+1
                v = j * dv
                P_temp[n, j] = P[n+1, j+1] * A1 * v * s + P[n+1, j] * (A2 * v * s**2 + r * s * A3) + P[n+1, j-1] * (-A1 * v * s) + P[n, j+1] * (v * A4 + (kappa * (theta - v) * dt)/(2 * dv)) + P[n, j] * (1 - 2 * A2 * v * s**2 - 2 * A4 * v - r * dt) + P[n, j-1] * (v * A4 - (kappa * (theta - v) * dt)/(2 * dv)) + P[n-1, j+1] * (-A1 * v * s) + P[n-1, j] * (A2 * v * s**2 - r * s * A3) + P[n-1, j-1] * A1 * v * s
        P = P_temp
    
    #for n in range(N-2):
        #P_0[n+1, 1:(J-2)] = P
    P_0 = P
    return P_0







# Example run
params = dict(
    r=0.05, kappa=1.0, theta=0.2,
    eta=0.5, rho=-0.4,
    K=100, T=1.0, S0=100, V0=0.25,
    S_max=300, V_max=1.0)

results = []
for (dt, ds, dv) in [(0.000005, 2, 0.125), (0.0005, 20, 0.125), (0.0005, 2, 0.125), (0.00005, 2, 0.125)]:
    print('I am running')
    price = heston_explicit(**params, dt = dt, ds = ds, dv = dv)
    results.append((params['T']/dt, ds, dv, price))
    print(f"The computed price at S0={S0}, V0={V0}, with numercal steps dt={dt}, ds={ds} and dv={dv} is given by {price[int(100/ds), int(0.25/dv)]}")

# display as table
#import pandas as pd
#df = pd.DataFrame(results, columns=["dt","N_S","N_V","FD price"])
#df["MC benchmark"] = 9.05  # replace with your Asn 1 MC result
#print(df)
