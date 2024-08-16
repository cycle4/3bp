from scipy.integrate import solve_ivp
import numpy as np

import sys
sys.path.append('./')
from cr3bp import cr3bp_alpha_mod
from init_values import init_constants, init_values_opt, init_char_values


def nonlcon(X0):

    # Ensure X0 is at least 2-dimensional
    X0 = np.atleast_2d(X0)
    print(X0)
    # Initialize 'c' as a NumPy array with zeros, matching the MATLAB function structure
    c = np.zeros((len(X0), 2))
    
    # Iterate over each row in X0
    for i in range(X0.shape[0]):
        c[i, 0] = np.linalg.norm(X0[i, 0]) - 1  # Equivalent of norm(X0(i,1)) - 1
        c[i, 1] = X0[i, 1] - np.pi              # Equivalent of X0(i,2) - pi
    
    ceq = np.array([])  # ceq is an empty array, equivalent to ceq = []
    return c, ceq


# def nonlcon(X0):
#     # Initialize constants
#     G, m_earth, m_moon, mu, r_earth, r_orbit, mu_earth, n, g0 = init_constants()
#     # Initialize characteristic values
#     lstar, mstar, mu_star, tstar, vstar, astar = init_char_values(G, m_earth, m_moon)
#     # Initialize other values
#     s0, sd, thrust_max_nd, m_dot_max_nd, N_step, time_of_flight, tspan, m_min = init_values_opt()
#     # Options for solve_ivp
#     options = {'rtol': 1e-12, 'atol': 1e-12}
#     # Define the control vector
#     uref = 1 - 58568e-10  # u0

#     # Perform the integration using solve_ivp (equivalent to ode45 in MATLAB)
#     sol = solve_ivp(lambda t, x: cr3bp_alpha_mod(t, x, mu, n, thrust_max_nd, m_dot_max_nd, uref),
#                     (tspan[0], tspan[-1]), s0, method='RK45', t_eval=tspan, **options)
#     ti = sol.t
#     si = sol.y.T  # Transpose to match MATLAB's output shape

#     dist = np.linalg.norm(si[:, :3] - sd[:3], axis=1)
#     distmin_idx = np.argmin(dist)

#     max_position_error = np.linalg.norm(si[distmin_idx, 3:6]) * time_of_flight / N_step
#     max_velocity_error = np.linalg.norm(sd[3:6]) * 0.01

#     # Finding the appropriate state and time
#     for i in range(len(si[:, 0])):
#         if (abs(si[i, 0] - sd[0]) <= max_position_error and
#             abs(si[i, 1] - sd[1]) <= max_position_error and
#             abs(si[i, 3] - sd[3]) <= max_velocity_error and
#             abs(si[i, 4] - sd[4]) <= max_velocity_error):
            
#             state = si[:i+1, :]
#             time = ti[:i+1]
            
#             dist_state = np.linalg.norm(state[:, :3] - sd[:3], axis=1)
#             distmin_idx_state = np.argmin(dist_state)
#             break
#     else:
#         state = si
#         time = ti
#         dist_state = np.linalg.norm(state[:, :3] - sd[:3], axis=1)
#         distmin_idx_state = np.argmin(dist_state)

#     # Defining inequality constraints
#     Ce = [
#         abs(sd[0] - state[distmin_idx_state, 0]) - max_position_error,
#         abs(sd[1] - state[distmin_idx_state, 1]) - max_position_error,
#         m_min - state[-1, 6]
#     ]

#     # No equality constraints
#     Ceq = []

#     return Ce, Ceq