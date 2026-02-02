import numpy as np
import matplotlib.pyplot as plt
from transforms3d.quaternions import qmult, qnorm
from transforms3d.euler import mat2euler
import argparse
from load_data import load_dataset
from IMU_calibration import calibrate_imu
from helpers import vicon_to_ts_Rmats, quat_traj_to_Rmats, past_index

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="1",
        help="Dataset number as string"
    )

    parser.add_argument(
        "--T_STATIC",
        type=float,
        default=5.0,
        help="Static interval duration (seconds)"
    )

    parser.add_argument(
        "--ADC_VREF",
        type=float,
        default=3.3,
        help="ADC reference voltage (V)"
    )

    parser.add_argument(
        "--ADC_RES",
        type=int,
        default=1023,
        help="ADC resolution (e.g. 1023 for 10-bit)"
    )

    parser.add_argument(
        "--acc_sens_mV_per_g",
        type=float,
        default=300.0,
        help="Accelerometer sensitivity (mV/g)"
    )

    parser.add_argument(
        "--gyro_sens_mV_per_deg_s_4x",
        type=float,
        default=3.33,
        help="Gyro sensitivity at 4x gain"
    )

    return parser.parse_args()
def delta_quat_from_omega_dt(omega_rad_s, dt):
    rotvec = omega_rad_s * dt
    theta = np.linalg.norm(rotvec)

    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    axis = rotvec / theta
    half = 0.5 * theta
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float64)

def integrate_gyro_quat(ts, gyro_rad_s, q0=None):
    N = ts.shape[0]
    q = np.zeros((N, 4), dtype=np.float64)
    q[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) if q0 is None else np.array(q0, dtype=np.float64)

    for k in range(N - 1):
        dt = float(ts[k + 1] - ts[k])
        if dt <= 0:
            q[k + 1] = q[k]
            continue
        dq = delta_quat_from_omega_dt(gyro_rad_s[:, k], dt)
        q[k + 1] = qmult(q[k], dq)
        q[k + 1] = q[k + 1] / qnorm(q[k + 1])

    return q

def rot_angle_from_R(R):

    c = 0.5 * (np.trace(R) - 1.0)
    c = np.clip(c, -1.0, 1.0)
    return float(np.arccos(c))


def so3_angle_errors(R_vicon_on_imu, R_imu):
    N = R_imu.shape[0]
    theta = np.zeros(N, dtype=np.float64)
    for k in range(N):
        Rerr = R_vicon_on_imu[k].T @ R_imu[k]
        theta[k] = rot_angle_from_R(Rerr)
    return theta

def rmats_to_euler(Rmats, axes="sxyz"):
    N = Rmats.shape[0]
    rpy = np.zeros((N, 3), dtype=np.float64)
    for k in range(N):
        rpy[k] = mat2euler(Rmats[k], axes=axes)
    rpy = np.unwrap(rpy, axis=0)
    return rpy[:, 0], rpy[:, 1], rpy[:, 2]


def main():

    args = parse_args()

    dataset = args.dataset
    if (dataset == "1" or dataset == "2" or dataset == "8" or dataset == "9"):
        camd, imud, vicd = load_dataset(dataset)
    else:
        imud, vicd = load_dataset(dataset)

    ts, acc_g_cal, gyro_rad_s, info = calibrate_imu(
        imud,
        T_STATIC=args.T_STATIC,
        ADC_VREF=args.ADC_VREF,
        ADC_RES=args.ADC_RES,
        acc_sens_mV_per_g=args.acc_sens_mV_per_g,
        gyro_sens_mV_per_deg_s_4x=args.gyro_sens_mV_per_deg_s_4x,
    )

    print("\nCalibration info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Gyro-only integration
    q_traj = integrate_gyro_quat(ts, gyro_rad_s, q0=[1, 0, 0, 0])
    R_imu = quat_traj_to_Rmats(q_traj)

    # VICON rotations
    ts_v, R_vicon = vicon_to_ts_Rmats(vicd)

    idx = past_index(ts_v, ts)
    R_v_on_imu = R_vicon[idx]

    theta = so3_angle_errors(R_v_on_imu, R_imu)

    theta_rms = float(np.sqrt(np.mean(theta ** 2)))
    theta_rms_deg = theta_rms * 180.0 / np.pi

    print("\nSO(3) rotation-angle error:")
    print("  RMS (rad):", theta_rms)
    print("  RMS (deg):", theta_rms_deg)

    t = ts - ts[0]

    roll_i, pitch_i, yaw_i = rmats_to_euler(R_imu, axes="sxyz")
    roll_v, pitch_v, yaw_v = rmats_to_euler(R_v_on_imu, axes="sxyz")

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axs[0].plot(t, roll_v, label="VICON", linewidth=2)
    axs[0].plot(t, roll_i, label="IMU gyro-integrated", alpha=0.9)
    axs[0].set_ylabel("Roll (rad)")
    axs[0].legend()

    axs[1].plot(t, pitch_v, label="VICON", linewidth=2)
    axs[1].plot(t, pitch_i, label="IMU gyro-integrated", alpha=0.9)
    axs[1].set_ylabel("Pitch (rad)")
    axs[1].legend()

    axs[2].plot(t, yaw_v, label="VICON", linewidth=2)
    axs[2].plot(t, yaw_i, label="IMU gyro-integrated", alpha=0.9)
    axs[2].set_ylabel("Yaw (rad)")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend()

    plt.suptitle(f"Dataset {dataset}: Gyro-only integration vs VICON (axes='sxyz')")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
