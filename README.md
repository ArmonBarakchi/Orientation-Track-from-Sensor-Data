# How to Use

This project contains the scripts relating to PR1 for ECE276A. Below, there is a guideline for how to run each piece of code.
## Setup

Before running, do one of the follow

### option 1
```bash
pip install -r requirements.txt
```
### option2
```bash
conda env create -f environment.yaml
conda activate <environment-name>
```

Ensure that the <environment-name> matches the environment name in the .yaml file


## IMU calibration

```bash
python IMU_calibration.py [OPTIONS]
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | string | "1" | Dataset number as string |
| `--T_STATIC` | float | 5.0 | Static interval duration in seconds |
| `--ADC_VREF` | float | 3.3 | ADC reference voltage in volts |
| `--ADC_RES` | int | 1023 | ADC resolution, e.g. 1023 for 10-bit |
| `--acc_sens_mV_per_g` | float | 300.0 | Accelerometer sensitivity in mV/g |
| `--gyro_sens_mV_per_deg_s_4x` | float | 3.33 | Gyro sensitivity at 4x gain |

### Example

```bash
python imu_calibration.py --dataset "2" --T_STATIC 6.0
```

---

## Projected Gradient Descent Orientation

```bash
python pgd_orientation.py [OPTIONS]
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | choice | "train" | Run mode: `train` (uses VICON comparisons) or `test` (no VICON) |
| `--dataset` | string | "1" | Dataset id as string; train: 1-9, test: 10,11 |
| `--T_STATIC` | float | 5.0 | Seconds used for static bias estimation |
| `--ADC_VREF` | float | 3.3 | ADC reference voltage |
| `--ADC_RES` | int | 1024 | ADC resolution levels, e.g. 1024 for 10-bit |
| `--acc_sens_mV_per_g` | float | 300.0 | Accelerometer sensitivity in mV/g |
| `--gyro_sens_mV_per_deg_s_4x` | float | 3.33 | Gyro sensitivity 4x in mV/(deg/s) |
| `--lr` | float | 1e-3 | Learning rate for projected gradient descent |
| `--iters` | int | 1000 | Iterations for projected gradient descent |

### Example

```bash
python pgd_orientation.py --mode train --dataset "5" --lr 0.001 --iters 1500
```

---

## Panorama Generation

```bash
python panorama.py [OPTIONS]
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | choice | "test" | `train` (expects VICON and camera) or `test` (no VICON) |
| `--dataset` | string | None | Dataset id as string |
| `--T_STATIC` | float | 5.0 | Static interval duration in seconds |
| `--ADC_VREF` | float | 3.3 | ADC reference voltage |
| `--ADC_RES` | int | 1024 | ADC resolution levels |
| `--acc_sens_mV_per_g` | float | 300.0 | Accelerometer sensitivity in mV/g |
| `--gyro_sens_mV_per_deg_s_4x` | float | 3.33 | Gyro sensitivity at 4x gain |
| `--lr` | float | 1e-3 | Learning rate for optimization |
| `--iters` | int | 1000 | Optimization iterations |
| `--projection` | choice | "cylindrical" | Projection type: `cylindrical` or `spherical` |
| `--focal_scale` | float | 0.4 | Focal length scale factor |

### Example

```bash
python panorama.py --mode test --dataset "10" --projection spherical --focal_scale 0.5
```

---

## Validate Calibration


```bash
python validateCalibration.py [OPTIONS]
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | string | "1" | Dataset number as string |
| `--T_STATIC` | float | 5.0 | Static interval duration in seconds |
| `--ADC_VREF` | float | 3.3 | ADC reference voltage in volts |
| `--ADC_RES` | int | 1023 | ADC resolution, e.g. 1023 for 10-bit |
| `--acc_sens_mV_per_g` | float | 300.0 | Accelerometer sensitivity in mV/g |
| `--gyro_sens_mV_per_deg_s_4x` | float | 3.33 | Gyro sensitivity at 4x gain |

### Example

```bash
python validate_calibration.py --dataset "3" --ADC_VREF 3.0
```

