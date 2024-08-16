import numpy as np

from loguru import logger
import numpy as np

def cr3bp_alpha_mod(t, s, mu, n, thrust_max_nd, m_dot_max_nd, u0):
    # Ensure u0 is scalar, if X0 is passed as an array, you should extract a scalar
    if isinstance(u0, np.ndarray) or isinstance(u0, list):
        u0 = u0[0]  # or another appropriate index or operation
    
    ds = np.zeros(7)

    norm_v = np.linalg.norm(s[3:6])
    if norm_v == 0:
        logger.error("Zero velocity encountered, cannot normalize.")
        raise ValueError("Zero velocity encountered, cannot normalize.")
    V1 = s[3:6] / norm_v

    x, y, z = s[0], s[1], s[2]
    vx, vy, vz = s[3], s[4], s[5]
    m = s[6]

    r = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    d = np.sqrt((x + mu)**2 + y**2 + z**2)

    ds[0] = vx
    ds[1] = vy
    ds[2] = vz

    ds[3] = (2 * n * vy + x 
             - mu * (x - 1 + mu) / r**3 
             - (1 - mu) * (x + mu) / d**3 
             + u0 * thrust_max_nd * V1[0] / m)
    ds[4] = (-2 * n * vx + y 
             - y * mu / r**3 
             - y * (1 - mu) / d**3 
             + u0 * thrust_max_nd * V1[1] / m)
    ds[5] = (-z * mu / r**3 
             - z * (1 - mu) / d**3 
             + u0 * thrust_max_nd * V1[2] / m)

    ds[6] = -m_dot_max_nd * u0

    return ds
