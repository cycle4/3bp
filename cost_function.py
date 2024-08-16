import numpy as np
from scipy.integrate import solve_ivp
from loguru import logger

import sys
sys.path.append('./')
from cr3bp import cr3bp_alpha_mod
from init_values import init_constants, init_values_opt


def cost_function(X0):
    # Define constants and initial values
    _, _, _, mu, _, _, _, n, _ = init_constants()
    s0, sd, thrust_max_nd, m_dot_max_nd, _, _, tspan, _ = init_values_opt()

    # Define the ODE function to be passed to solve_ivp
    def cr3bp_alpha(t, x):
        return cr3bp_alpha_mod(t, x, mu, n, thrust_max_nd, m_dot_max_nd, X0)

    # Extract the start and end times from tspan
    t0, tf = tspan[0], tspan[-1]

    # Solve the ODE using solve_ivp
    sol = solve_ivp(cr3bp_alpha, (t0, tf), s0, method='RK45', t_eval=tspan, rtol=1e-12, atol=1e-12)

    # Calculate the distance
    dist = np.linalg.norm(sol.y[:3, :] - sd[:3, np.newaxis], axis=0)
    # Find the index of minimum distance
    distmin_idx = np.argmin(dist)
    logger.success("Distance : {} at time : {}", dist[distmin_idx], sol.t[distmin_idx])


    tof = sol.t[distmin_idx]

    # Weight constants
    W1 = 20
    W2 = 2
    W3 = 1e5

    m_final_nd = sol.y[6, -1]

    sdv = sd[3:6]
    sfv = sol.y[3:6, distmin_idx]

    # Calculate the cost function
    Jfun = -0.5 * W1 * (1 - m_final_nd) + W2 * tof + W3 * np.linalg.norm(sdv - sfv)

    return Jfun


