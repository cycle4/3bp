import numpy as np
from scipy.integrate import solve_ivp

def global_integrate(alpha, s0, N_int, N_step, tspan, X0, mu, n, thrust_max_nd, m_dot_max_nd):
    tol = 1e-12  # Convergence tolerance for ODE solver
    sf = np.zeros((N_step * N_int + 1, 7))  # Initialize solution array
    t = np.zeros(N_step * N_int + 1)  # Initialize time array

    for i in range(N_int):
        u0 = alpha[3*(i):3*(i+1)]
        u = np.linalg.norm(u0)
        
        if i == 0:
            tspan_i = tspan[0:N_step+1]
        elif 0 < i < N_int - 1:
            tspan_i = tspan[(i-1) * N_step:i * N_step + 1]
        else:
            tspan_i = tspan[(i-1) * N_step:i * N_step + 2]
        
        sol = solve_ivp(
            lambda t, x: CR3BP_alpha(t, x, mu, n, thrust_max_nd, X0, m_dot_max_nd),
            tspan_i, s0, method='RK45', rtol=tol, atol=tol
        )

        if i == 0:
            sf[0:N_step+1, :] = sol.y.T[0:N_step+1, :]
            t[0:N_step+1] = sol.t[0:N_step+1]
        elif 0 < i < N_int - 1:
            sf[(i-1) * N_step:i * N_step + 1, :] = sol.y.T[0:N_step+1, :]
            t[(i-1) * N_step:i * N_step + 1] = sol.t[0:N_step+1]
        else:
            sf[(i-1) * N_step:i * N_step + 2, :] = sol.y.T[0:N_step+2, :]
            t[(i-1) * N_step:i * N_step + 2] = sol.t[0:N_step+2]

        s0 = sol.y[:, -1]  # Update initial condition for next integration

    return t, sf
