import numpy as np

def vec2SkewSymmetricMat(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0 , -v[0]],
                     [-v[1], v[0], 0]])


def pinv(J, theta=0.03, damping=0.2, use_damping=True):
    U, s, V_t = np.linalg.svd(J)

    if not use_damping:
        s_d = np.zeros(s.shape)
        mask = s>=theta
        s_d[mask] = 1/s[mask]
        s_d[np.bitwise_not(mask)] = np.divide(s[np.bitwise_not(mask)],theta**2)
    else:
        s_d = np.divide(s, np.multiply(s, s) + damping**2)
    S_d = np.zeros(J.shape)
    S_d[:s_d.shape[0], :s_d.shape[0]] = np.diag(s_d)

    return V_t.T @ S_d.T @ U.T
