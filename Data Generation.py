"""
data_generation.py
Case 1: 1-D steady diffusion equation

    d/dx ( k dP/dx ) = f,     0 <= x <= 1
    P(0) = 0,  P(1) = 0

For Case 1, k and f are taken as CONSTANT for each realization, but
their values are randomly sampled from prescribed ranges.

For constant k and f:
    k * d2P/dx2 = f

and the exact solution is:
    P(x) = f/(2k) * (x^2 - x)

The generated data are saved in:
    case1_deeponet_data.npz

The branch input contains:
    [k(x_1), ..., k(x_m), f(x_1), ..., f(x_m)]

Since k and f are constant in Case 1, these are repeated values.
This format is intentionally kept compatible with a future extension
to spatially varying k(x) and f(x).
"""

import numpy as np
from pathlib import Path


# ----------------------------- USER SETTINGS ----------------------------- #
N_TRAIN = 2000
N_TEST = 400

N_SENSORS = 50          # Number of branch/training sensors
N_QUERY = 101           # Number of x locations for the pressure solution

K_MIN, K_MAX = 0.5, 2.0
F_MIN, F_MAX = 0.5, 2.0

SEED = 42
OUTPUT_FILE = "case1_deeponet_data.npz"
# ------------------------------------------------------------------------- #


def exact_pressure(x, k, f):
    """Exact Case-1 pressure solution."""
    return (f / (2.0 * k)) * (x**2 - x)


def generate_case1_data(n_samples, x_sensors, x_query, rng):
    """
    Generate branch inputs and corresponding pressure fields.

    Returns
    -------
    branch_input : (n_samples, 2*N_SENSORS)
        First half = k at sensors
        Second half = f at sensors
    pressure : (n_samples, N_QUERY)
        Exact pressure solution at query points
    k_values : (n_samples,)
    f_values : (n_samples,)
    """
    k_values = rng.uniform(K_MIN, K_MAX, size=n_samples)
    f_values = rng.uniform(F_MIN, F_MAX, size=n_samples)

    # Constant fields sampled at the branch sensors.
    k_fields = np.repeat(k_values[:, None], len(x_sensors), axis=1)
    f_fields = np.repeat(f_values[:, None], len(x_sensors), axis=1)

    branch_input = np.concatenate([k_fields, f_fields], axis=1)

    pressure = np.empty((n_samples, len(x_query)), dtype=np.float32)

    for i in range(n_samples):
        pressure[i] = exact_pressure(
            x_query, k_values[i], f_values[i]
        ).astype(np.float32)

    return (
        branch_input.astype(np.float32),
        pressure,
        k_values.astype(np.float32),
        f_values.astype(np.float32),
    )


def main():
    rng = np.random.default_rng(SEED)

    # Sensors used by the branch network.
    x_sensors = np.linspace(0.0, 1.0, N_SENSORS, dtype=np.float32)

    # Locations at which we want the pressure field.
    x_query = np.linspace(0.0, 1.0, N_QUERY, dtype=np.float32)

    train_branch, train_pressure, train_k, train_f = generate_case1_data(
        N_TRAIN, x_sensors, x_query, rng
    )

    test_branch, test_pressure, test_k, test_f = generate_case1_data(
        N_TEST, x_sensors, x_query, rng
    )

    output_path = Path(__file__).resolve().parent / OUTPUT_FILE

    np.savez_compressed(
        output_path,
        x_sensors=x_sensors,
        x_query=x_query,
        train_branch=train_branch,
        train_pressure=train_pressure,
        train_k=train_k,
        train_f=train_f,
        test_branch=test_branch,
        test_pressure=test_pressure,
        test_k=test_k,
        test_f=test_f,
    )

    print("=" * 65)
    print("Case 1 DeepONet data generation completed.")
    print("=" * 65)
    print(f"Saved file : {output_path}")
    print(f"Training samples : {N_TRAIN}")
    print(f"Testing samples  : {N_TEST}")
    print(f"Number of sensors: {N_SENSORS}")
    print(f"Query points     : {N_QUERY}")
    print(f"k range          : [{K_MIN}, {K_MAX}]")
    print(f"f range          : [{F_MIN}, {F_MAX}]")
    print()
    print("Branch input shape :", train_branch.shape)
    print("Pressure shape     :", train_pressure.shape)


if __name__ == "__main__":
    main()
