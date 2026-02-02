import argparse
import cv2
import torch
from transforms3d.euler import  quat2mat
from load_data import load_dataset
from IMU_calibration import calibrate_imu
import numpy as np
from helpers import gyro_init_quats, quat_traj_to_Rmats, past_index
from pgd_orientation import projected_gd
def parse_args():
    p = argparse.ArgumentParser()

    # mode + dataset
    p.add_argument("--mode", choices=["train", "test"], default="test",
                   help="train: expects VICON and cam. test: no VICON.")
    p.add_argument("--dataset", type=str, default=None,
                   help="Dataset id as string")

    # calibration
    p.add_argument("--T_STATIC", type=float, default=5.0)
    p.add_argument("--ADC_VREF", type=float, default=3.3)
    p.add_argument("--ADC_RES", type=int, default=1024)
    p.add_argument("--acc_sens_mV_per_g", type=float, default=300.0)
    p.add_argument("--gyro_sens_mV_per_deg_s_4x", type=float, default=3.33)

    # PGD
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--iters", type=int, default=1000)

    # panorama
    p.add_argument("--projection", choices=["cylindrical", "spherical"], default="cylindrical")
    p.add_argument("--focal_scale", type=float, default=0.4)

    return p.parse_args()

def quat_traj_to_Rmats(q_traj: np.ndarray) -> np.ndarray:
    N = q_traj.shape[0]
    R = np.zeros((N, 3, 3), dtype=np.float64)
    for i in range(N):
        R[i] = quat2mat(q_traj[i])
    return R


#PANORAMA FUNCTIONS
def precompute_bearings(K: np.ndarray, W: int, H: int) -> np.ndarray:

    u, v = np.meshgrid(np.arange(W), np.arange(H))

    ones = np.ones_like(u)
    p_hom = np.stack([u, v, ones], axis=-1)

    K_inv = np.linalg.inv(K)
    X_cam = np.einsum('ij,hwj->hwi', K_inv, p_hom)

    norms = np.linalg.norm(X_cam, axis=-1, keepdims=True)
    X_cam = X_cam / (norms + 1e-12)

    return X_cam

def project_spherical(y_world: np.ndarray) -> tuple:
    X = y_world[..., 0]
    Y = y_world[..., 1]
    Z = y_world[..., 2]

    theta = np.arctan2(X, Z)

    theta = (theta + 2 * np.pi) % (2 * np.pi)

    phi = np.arcsin(np.clip(Y, -1.0, 1.0))

    return theta, phi

def project_cylindrical(y_world: np.ndarray) -> tuple:

    X = y_world[..., 0]
    Y = y_world[..., 1]
    Z = y_world[..., 2]

    theta = np.arctan2(X, Z)
    theta = (theta + 2 * np.pi) % (2 * np.pi)

    v = Y
    return theta, v

def compute_panorama_bounds(theta_all, vert_all, margin=0.02):

    th_min = np.min(theta_all)
    th_max = np.max(theta_all)
    v_min = np.min(vert_all)
    v_max = np.max(vert_all)

    th_range = th_max - th_min
    v_range = v_max - v_min

    th_min -= margin * th_range
    th_max += margin * th_range
    v_min -= margin * v_range
    v_max += margin * v_range

    return th_min, th_max, v_min, v_max

def rasterize_panorama(I_rgb: np.ndarray, U: np.ndarray, V: np.ndarray,
                       pano: np.ndarray, weight: np.ndarray = None):

    Hp, Wp = pano.shape[:2]

    U_int = np.round(U).astype(np.int32)
    V_int = np.round(V).astype(np.int32)

    valid = (U_int >= 0) & (U_int < Wp) & (V_int >= 0) & (V_int < Hp)

    v_idx, u_idx = np.where(valid)

    if weight is not None:
        for i in range(len(v_idx)):
            pv, pu = v_idx[i], u_idx[i]
            pano_v = V_int[pv, pu]
            pano_u = U_int[pv, pu]

            pano[pano_v, pano_u] += I_rgb[pv, pu].astype(np.float32)
            weight[pano_v, pano_u] += 1.0
    else:
        for i in range(len(v_idx)):
            pv, pu = v_idx[i], u_idx[i]
            pano_v = V_int[pv, pu]
            pano_u = U_int[pv, pu]

            pano[pano_v, pano_u] = I_rgb[pv, pu].astype(np.float32)

def build_panorama(camd, ts_imu, R_WB, K, R_BO=None, projection="spherical",
                   pano_size=None, use_average=False):

    if R_BO is None:
        R_BO = np.eye(3)

    cam_ts = np.asarray(camd["ts"]).reshape(-1)
    cam_imgs = camd["cam"]


    cam_imgs = np.transpose(cam_imgs, (3, 0, 1, 2))


    Kimgs, H, W, _ = cam_imgs.shape

    X_cam = precompute_bearings(K, W, H)
    ref_thetas = []

    u0 = W // 2
    v0 = H // 2
    x0 = X_cam[v0, u0]

    for k in range(Kimgs):
        i = past_index(ts_imu, cam_ts[k])
        R_WO = R_WB[i] @ R_BO
        y0 = R_WO @ x0

        theta0 = np.arctan2(y0[0], y0[2])
        ref_thetas.append(theta0)

    ref_thetas = np.unwrap(np.array(ref_thetas))

    theta_list = []
    vert_list = []

    for k in range(Kimgs):
        i = past_index(ts_imu, cam_ts[k])
        R_WO = R_WB[i] @ R_BO


        Y_world = np.einsum('ij,hwj->hwi', R_WO, X_cam)

        if projection == "spherical":
            theta, vert = project_spherical(Y_world)
        elif projection == "cylindrical":
            theta, vert = project_cylindrical(Y_world)
            theta += (ref_thetas[k] - np.mean(theta))
        else:
            raise ValueError("projection must be 'spherical' or 'cylindrical'")

        theta_list.append(theta)
        vert_list.append(vert)

    theta_all = np.concatenate([t.reshape(-1) for t in theta_list])
    vert_all = np.concatenate([v.reshape(-1) for v in vert_list])

    if pano_size is None:
        th_min, th_max, v_min, v_max = compute_panorama_bounds(theta_all, vert_all)

        pixels_per_radian = 300
        Wp = int(np.ceil((th_max - th_min) * pixels_per_radian))
        Hp = int(np.ceil((v_max - v_min) * pixels_per_radian))

        Wp = max(100, min(Wp, 4000))
        Hp = max(100, min(Hp, 2000))
    else:
        Hp, Wp = pano_size
        th_min, th_max, v_min, v_max = compute_panorama_bounds(theta_all, vert_all)

    pano = np.zeros((Hp, Wp, 3), dtype=np.float32)
    weight = np.zeros((Hp, Wp), dtype=np.float32) if use_average else None

    for k in range(Kimgs):
        i = past_index(ts_imu, cam_ts[k])
        R_WO = R_WB[i] @ R_BO

        Y_world = np.einsum('ij,hwj->hwi', R_WO, X_cam)  # (H,W,3)

        if projection == "spherical":
            theta, vert = project_spherical(Y_world)
        elif projection == "cylindrical":
            theta, vert = project_cylindrical(Y_world)
        else:
            raise ValueError("projection must be 'spherical' or 'cylindrical'")

        U = (theta - th_min) / (th_max - th_min) * (Wp - 1)
        V = (vert - v_min) / (v_max - v_min) * (Hp - 1)

        rasterize_panorama(cam_imgs[k], U, V, pano, weight)

    if use_average:
        valid_mask = weight > 0
        pano[valid_mask] /= weight[valid_mask, np.newaxis]

    return np.clip(pano, 0, 255).astype(np.uint8)


def quat_traj_to_Rmats(q_traj: np.ndarray) -> np.ndarray:
    N = q_traj.shape[0]
    R = np.zeros((N, 3, 3), dtype=np.float64)
    for i in range(N):
        R[i] = quat2mat(q_traj[i])
    return R


def main():
    args = parse_args()
    mode = args.mode
    dataset = args.dataset if args.dataset is not None else ("8" if mode == "train" else "11")

    R_BO = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=np.float64)
    R_BO = R_BO.T

    if(mode == "train"):
        if dataset in {"1", "2", "8", "9"}:
            camd, imud, vicd = load_dataset(dataset)
        else:
            raise ValueError("There is no cam data for your selected dataset.")

        ts, acc_g_cal, gyro_rad_s, info = calibrate_imu(
            imud,
            T_STATIC=args.T_STATIC,
            ADC_VREF=args.ADC_VREF,
            ADC_RES=args.ADC_RES,
            acc_sens_mV_per_g=args.acc_sens_mV_per_g,
            gyro_sens_mV_per_deg_s_4x=args.gyro_sens_mV_per_deg_s_4x,
        )

        q_init_np = gyro_init_quats(ts, gyro_rad_s)

        dt = (ts[1:] - ts[:-1]).astype(np.float64)
        omega = gyro_rad_s[:, :-1].T.astype(np.float64)
        acc = acc_g_cal.T.astype(np.float64)

        Q0 = torch.tensor(q_init_np, dtype=torch.float32)
        dt_t = torch.tensor(dt, dtype=torch.float32)
        omega_t = torch.tensor(omega, dtype=torch.float32)
        acc_t = torch.tensor(acc, dtype=torch.float32)

        Q_est = projected_gd(Q0, dt_t, omega_t, acc_t, lr=args.lr, iters=args.iter)
        q_est_np = Q_est.detach().cpu().numpy()

        R_est = quat_traj_to_Rmats(q_est_np)

        ts_imu = ts

        R_WB = np.linalg.inv(R_est)
        H, W = camd["cam"].shape[0], camd["cam"].shape[1]
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        f = args.focal_scale*max(W,H)

        K = np.array([
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        pano = build_panorama(camd, ts_imu, R_WB, K, R_BO=R_BO, projection=args.projection)
        cv2.imshow('Panorama', cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if (mode == "test"):

        if dataset in {"10", "11"}:
            camd, imud = load_dataset(dataset)
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

        q_init_np = gyro_init_quats(ts, gyro_rad_s)

        dt = (ts[1:] - ts[:-1]).astype(np.float64)
        omega = gyro_rad_s[:, :-1].T.astype(np.float64)
        acc = acc_g_cal.T.astype(np.float64)

        Q0 = torch.tensor(q_init_np, dtype=torch.float32)
        dt_t = torch.tensor(dt, dtype=torch.float32)
        omega_t = torch.tensor(omega, dtype=torch.float32)
        acc_t = torch.tensor(acc, dtype=torch.float32)

        Q_est = projected_gd(Q0, dt_t, omega_t, acc_t, lr=args.lr, iters=args.iters)
        q_est_np = Q_est.detach().cpu().numpy()

        R_est = quat_traj_to_Rmats(q_est_np)
        ts_imu = ts

        R_WB = np.linalg.inv(R_est)
        H, W = camd["cam"].shape[0], camd["cam"].shape[1]
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        f = args.focal_scale * max(W, H)

        K = np.array([
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        pano = build_panorama(camd, ts_imu, R_WB, K, R_BO=R_BO, projection=args.projection)
        cv2.imshow('Panorama', cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
