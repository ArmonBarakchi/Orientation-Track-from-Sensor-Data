
import numpy as np
from transforms3d.quaternions import qmult, qnorm, quat2mat
def vicon_to_ts_Rmats(vicd):

    ts_v = np.asarray(vicd["ts"], dtype=np.float64).reshape(-1)
    Rraw = np.asarray(vicd["rots"], dtype=np.float64)
    Rmats = np.transpose(Rraw, (2, 0, 1))

    return ts_v, Rmats

def delta_quat_from_omega_dt_np(omega_rad_s, dt):
    rotvec = omega_rad_s * dt
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rotvec / theta
    half = 0.5 * theta
    return np.array([np.cos(half), *(axis * np.sin(half))], dtype=np.float64)

def gyro_init_quats(ts, gyro_rad_s):
    N = ts.shape[0]
    q = np.zeros((N, 4), dtype=np.float64)
    q[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for k in range(N - 1):
        dt = float(ts[k + 1] - ts[k])
        dq = delta_quat_from_omega_dt_np(gyro_rad_s[:, k], dt)
        q[k + 1] = qmult(q[k], dq)
        q[k + 1] = q[k + 1] / qnorm(q[k + 1])
    return q

def quat_traj_to_Rmats(q_traj):
    N = q_traj.shape[0]
    R = np.zeros((N, 3, 3), dtype=np.float64)
    for k in range(N):
        R[k] = quat2mat(q_traj[k])
    return R

def past_index(source_t, target_t):
    idx = np.searchsorted(source_t, target_t, side="right") - 1
    return np.clip(idx, 0, len(source_t) - 1)