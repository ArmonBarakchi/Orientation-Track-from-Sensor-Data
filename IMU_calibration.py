import numpy as np
from load_data import load_dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="1",
        help="Dataset number as string (e.g. '1', '2', '8', etc.)"
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

def calibrate_imu(
    imud: np.ndarray,
    T_STATIC: float = 2.0,
    ADC_VREF: float = 3.3,
    ADC_RES: int = 1023,
    acc_sens_mV_per_g: float = 300.0,
    gyro_sens_mV_per_deg_s_4x: float = 3.33,
):

    ts = imud[0, :].astype(np.float64)
    acc_raw = imud[1:4, :].astype(np.float64)
    gyro_raw = imud[4:7, :].astype(np.float64)

    static_idx = ts <= (ts[0] + T_STATIC)
    n_static = int(static_idx.sum())

    gyro_bias_raw = np.mean(gyro_raw[:, static_idx], axis=1)
    gyro_raw_cal = gyro_raw - gyro_bias_raw[:, None]

    V_per_count = ADC_VREF / float(ADC_RES)

    acc_sens_V_per_g = acc_sens_mV_per_g * 1e-3
    gyro_sens_V_per_deg_s = gyro_sens_mV_per_deg_s_4x * 1e-3

    acc_scale_g_per_count = V_per_count / acc_sens_V_per_g
    gyro_scale_deg_s_per_count = V_per_count / gyro_sens_V_per_deg_s
    gyro_scale_rad_s_per_count = gyro_scale_deg_s_per_count * (np.pi / 180.0)

    acc_g = acc_raw * acc_scale_g_per_count
    gyro_rad_s = gyro_raw_cal * gyro_scale_rad_s_per_count

    acc_static_mean = np.mean(acc_g[:, static_idx], axis=1)   # (3,)
    acc_bias_g = acc_static_mean - np.array([0.0, 0.0, 1.0])
    acc_g_cal = acc_g - acc_bias_g[:, None]

    gyro_static_mag = float(np.linalg.norm(gyro_rad_s[:, static_idx], axis=0).mean())
    acc_static_mag = float(np.linalg.norm(acc_g_cal[:, static_idx], axis=0).mean())

    info = {
        "T_STATIC": T_STATIC,
        "n_static": n_static,
        "gyro_bias_raw": gyro_bias_raw,
        "acc_bias_g": acc_bias_g,
        "acc_scale_g_per_count": acc_scale_g_per_count,
        "gyro_scale_rad_s_per_count": gyro_scale_rad_s_per_count,
        "gyro_static_mag_mean_rad_s": gyro_static_mag,
        "acc_static_mag_mean_g": acc_static_mag,
    }

    return ts, acc_g_cal, gyro_rad_s, info




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


if __name__ == "__main__":
    main()




