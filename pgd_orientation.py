import argparse
import matplotlib.pyplot as plt
import torch
from transforms3d.euler import quat2euler, mat2euler, quat2mat
from load_data import load_dataset
from IMU_calibration import calibrate_imu
import numpy as np
from helpers import vicon_to_ts_Rmats, gyro_init_quats, past_index


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["train", "test"], default="train",
                   help="Run mode: train uses VICON comparisons, test has no VICON")

    p.add_argument("--dataset", type=str, default="1",
                   help="Dataset id as a string (train: 1-9, test: 10,11")

    # IMU calibration params
    p.add_argument("--T_STATIC", type=float, default=5.0, help="Seconds used for static bias estimation")
    p.add_argument("--ADC_VREF", type=float, default=3.3, help="ADC reference voltage")
    p.add_argument("--ADC_RES", type=int, default=1024, help="ADC resolution (levels), e.g. 1024 for 10-bit")
    p.add_argument("--acc_sens_mV_per_g", type=float, default=300.0, help="Accelerometer sensitivity (mV/g)")
    p.add_argument("--gyro_sens_mV_per_deg_s_4x", type=float, default=3.33, help="Gyro sensitivity 4x (mV/(deg/s))")

    # Optimizer params
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate for projected GD")
    p.add_argument("--iters", type=int, default=1000, help="Iterations for projected GD")

    return p.parse_args()


def plot_rpy_single(ts, rpy, title="RPY"):
    t = ts - ts[0]
    labels = ["Roll", "Pitch", "Yaw"]

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for i in range(3):
        axs[i].plot(t, rpy[:, i], label="Estimate")   # <-- key change
        axs[i].set_ylabel(f"{labels[i]} (rad)")
        axs[i].legend()

    axs[-1].set_xlabel("Time (s)")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


#QUATERNION FUNCTIONS
def tqnormalize(q, eps=1e-12):
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + eps)

def tqconj(q):
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def tqinv(q, eps=1e-12):
    return tqconj(q) / (torch.sum(q*q, dim=-1, keepdim=True) + eps)

def tqmul(a, b):
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def qexp_pure(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)  # (...,1)
    w = torch.cos(v_norm)
    scale = torch.sin(v_norm) / (v_norm + eps)
    xyz = v * scale
    q = torch.cat([w, xyz], dim=-1)
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + eps)



def qlog_unit(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = q / (torch.linalg.norm(q, dim=-1, keepdim=True) + eps)
    w = q[..., 0:1]
    v = q[..., 1:4]
    v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    theta = torch.atan2(v_norm, w)
    scale = theta / (v_norm + eps)
    return torch.where(v_norm > eps, scale * v, v)



def gravity_body_pred(Q):
    g = torch.tensor([0.0, 0.0, 0.0, 1.0], device=Q.device, dtype=Q.dtype)
    g = g.expand(Q.shape[:-1] + (4,))
    return tqmul(tqmul(tqinv(Q), g), Q)[..., 1:]



def motion_predict(Q_t, dt, omega):
    v = 0.5 * dt[:, None] * omega
    dQ = qexp_pure(v)
    return tqnormalize(tqmul(Q_t, dQ))

def cost_function(Q, dt, omega, acc):
    Q_t = Q[:-1]
    Q_next = Q[1:]
    Q_pred_next = motion_predict(Q_t, dt, omega)

    delta = tqmul(tqinv(Q_next), Q_pred_next)
    delta = tqnormalize(delta)
    r_motion = 2.0 * qlog_unit(delta)
    motion_term = 0.5 * torch.sum(r_motion * r_motion)


    g_pred = gravity_body_pred(Q)
    r_obs = acc - g_pred
    obs_term = 0.5 * torch.sum(r_obs * r_obs)

    return motion_term + obs_term


def projected_gd(Q0, dt, omega, acc, lr=1e-4, iters=300):
    Q = Q0.clone().detach().requires_grad_(True)

    for k in range(iters):
        loss = cost_function(Q, dt, omega, acc)

        loss.backward()

        with torch.no_grad():
            Q -= lr * Q.grad
            Q[:] = tqnormalize(Q)
            Q.grad.zero_()

        if k % 25 == 0:
            print(f"iter {k:4d} | cost = {loss.detach().item():.6e}")

    return Q.detach()



def quat_traj_to_rpy(q_traj, axes="sxyz"):
    rpy = np.zeros((q_traj.shape[0], 3), dtype=np.float64)
    for i in range(q_traj.shape[0]):
        rpy[i] = quat2euler(q_traj[i], axes=axes)
    return np.unwrap(rpy, axis=0)

def rmats_to_rpy(Rmats, axes="sxyz"):
    rpy = np.zeros((Rmats.shape[0], 3), dtype=np.float64)
    for i in range(Rmats.shape[0]):
        rpy[i] = mat2euler(Rmats[i], axes=axes)
    return np.unwrap(rpy, axis=0)

def plot_rpy(ts, rpy_est, rpy_vicon, title):
    t = ts - ts[0]
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels = ["Roll", "Pitch", "Yaw"]
    for i in range(3):
        axs[i].plot(t, rpy_vicon[:, i], label="VICON", linewidth=2)
        axs[i].plot(t, rpy_est[:, i], label="Estimated", alpha=0.9)
        axs[i].set_ylabel(f"{labels[i]} (rad)")
        axs[i].legend()
    axs[-1].set_xlabel("Time (s)")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def quat_traj_to_Rmats(q_traj: np.ndarray) -> np.ndarray:
    N = q_traj.shape[0]
    R = np.zeros((N, 3, 3), dtype=np.float64)
    for i in range(N):
        R[i] = quat2mat(q_traj[i])
    return R

def rot_angle_from_R(R: np.ndarray) -> float:
    c = 0.5 * (np.trace(R) - 1.0)
    c = np.clip(c, -1.0, 1.0)
    return float(np.arccos(c))

def so3_angle_errors(R_gt: np.ndarray, R_est: np.ndarray) -> np.ndarray:
    N = R_est.shape[0]
    theta = np.zeros(N, dtype=np.float64)
    for i in range(N):
        Rerr = R_gt[i].T @ R_est[i]
        theta[i] = rot_angle_from_R(Rerr)
    return theta

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))

def report_so3_metrics(ts: np.ndarray, R_gt: np.ndarray, R_est: np.ndarray, label: str, plot: bool = True):
    theta = so3_angle_errors(R_gt, R_est)
    theta_rms = rms(theta)
    theta_rms_deg = theta_rms * 180.0 / np.pi

    print(f"\nSO(3) rotation-angle error ({label}):")
    print("  RMS (rad):", theta_rms)
    print("  RMS (deg):", theta_rms_deg)

    if plot:
        t = ts - ts[0]
        plt.figure(figsize=(12, 4))
        plt.plot(t, theta)
        plt.xlabel("Time (s)")
        plt.ylabel("Angle error (rad)")
        plt.title(f"{label}: SO(3) angle error")
        plt.tight_layout()
        plt.show()

    return theta, theta_rms, theta_rms_deg

def main():
    args = parse_args()
    mode = args.mode
    dataset = args.dataset

    if (mode == "train"):
        if dataset in {"1", "2", "8", "9"}:
            camd, imud, vicd = load_dataset(dataset)
        elif dataset in {"3", "4", "5", "6", "7"}:
            imud, vicd = load_dataset(dataset)
            camd = None
        else:
            raise ValueError(f"Unknown train dataset: {dataset}")

        ts, acc_g_cal, gyro_rad_s, info = calibrate_imu(
            imud,
            T_STATIC=args.T_STATIC,
            ADC_VREF=args.ADC_VREF,
            ADC_RES=args.ADC_RES,
            acc_sens_mV_per_g=args.acc_sens_mV_per_g,
            gyro_sens_mV_per_deg_s_4x=args.gyro_sens_mV_per_deg_s_4x,
        )


        # Gyro-only init
        q_init_np = gyro_init_quats(ts, gyro_rad_s)  # (N,4)

        # VICON (for plotting on train)
        ts_v, Rv = vicon_to_ts_Rmats(vicd)
        idx = past_index(ts_v, ts)
        Rv_on_imu = Rv[idx]

        R_init = quat_traj_to_Rmats(q_init_np)
        report_so3_metrics(ts, Rv_on_imu, R_init, label=f"Dataset {dataset}: gyro-only init", plot=True)

        rpy_init = quat_traj_to_rpy(q_init_np, axes="sxyz")
        rpy_v = rmats_to_rpy(Rv_on_imu, axes="sxyz")
        plot_rpy(ts, rpy_init, rpy_v, f"Dataset {dataset}: Gyro-only vs VICON")

        # Build tensors for PGD
        dt = (ts[1:] - ts[:-1]).astype(np.float64)               # (N-1,)
        omega = gyro_rad_s[:, :-1].T.astype(np.float64)          # (N-1,3)
        acc = acc_g_cal.T.astype(np.float64)                     # (N,3)

        Q0 = torch.tensor(q_init_np, dtype=torch.float32)
        dt_t = torch.tensor(dt, dtype=torch.float32)
        omega_t = torch.tensor(omega, dtype=torch.float32)
        acc_t = torch.tensor(acc, dtype=torch.float32)

        # Run PGD (WILL RAISE until you implement qexp_pure and qlog_unit)
        Q_est = projected_gd(Q0, dt_t, omega_t, acc_t, lr=args.lr, iters=args.iters)
        q_est_np = Q_est.detach().cpu().numpy()

        R_est = quat_traj_to_Rmats(q_est_np)
        report_so3_metrics(ts, Rv_on_imu, R_est, label=f"Dataset {dataset}: PGD estimate", plot=True)

        # Plot results
        rpy_est = quat_traj_to_rpy(q_est_np, axes="sxyz")
        plot_rpy(ts, rpy_est, rpy_v, f"Dataset {dataset}: PGD estimate vs VICON")

    elif mode == "test":
        if dataset in {"10", "11"}:
            camd, imud = load_dataset(dataset)
        else:
            raise ValueError(f"Unknown train dataset: {dataset}")

        # Calibrate IMU
        ts, acc_g_cal, gyro_rad_s, info = calibrate_imu(
            imud,
            T_STATIC=args.T_STATIC,
            ADC_VREF=args.ADC_VREF,
            ADC_RES=args.ADC_RES,
            acc_sens_mV_per_g=args.acc_sens_mV_per_g,
            gyro_sens_mV_per_deg_s_4x=args.gyro_sens_mV_per_deg_s_4x,
        )

        q_init_np = gyro_init_quats(ts, gyro_rad_s)  # (N,4)
        rpy_init = quat_traj_to_rpy(q_init_np, axes="sxyz")
        plot_rpy_single(ts, rpy_init, title=f"Test {dataset}: Gyro-only integration")

        # --- Build tensors for PGD ---
        dt = (ts[1:] - ts[:-1]).astype(np.float64)  # (N-1,)
        omega = gyro_rad_s[:, :-1].T.astype(np.float64)  # (N-1,3)
        acc = acc_g_cal.T.astype(np.float64)  # (N,3)

        Q0 = torch.tensor(q_init_np, dtype=torch.float32)
        dt_t = torch.tensor(dt, dtype=torch.float32)
        omega_t = torch.tensor(omega, dtype=torch.float32)
        acc_t = torch.tensor(acc, dtype=torch.float32)

        # --- Run PGD (optional but recommended for test plots) ---
        Q_est = projected_gd(Q0, dt_t, omega_t, acc_t, lr=args.lr, iters=args.iters)
        q_est_np = Q_est.detach().cpu().numpy()

        rpy_est = quat_traj_to_rpy(q_est_np, axes="sxyz")
        plot_rpy_single(ts, rpy_est, title=f"Test {dataset}: PGD estimate")
if __name__ == "__main__":
    main()
