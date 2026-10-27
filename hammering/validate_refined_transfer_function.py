"""Validate the transfer function identified from run 3 on run 4 unchanged."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PARAMETER_PATH = ROOT / "refined_time_domain_results" / "refined_modal_parameters.csv"
VALIDATION_CSV = ROOT / "成功4回目.CSV"
OUTPUT_DIR = ROOT / "refined_time_domain_results"
PRE_IMPACT_END_S = 0.025
DETECTION_SIGMA = 8.0
ANALYSIS_DURATION_S = 0.0554
IDENTIFIED_OFFSET_V = 0.0506876721


def read_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=20, header=None, encoding="utf-8-sig").iloc[:, :9]
    df.columns = ["number", "date", "clock_time", "time_us", "x_v", "y_v", "z_v", "alarm", "alarm_out"]
    for name in ["time_us", "x_v", "y_v", "z_v"]:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df.dropna(subset=["time_us", "x_v", "y_v", "z_v"]).reset_index(drop=True)
    df["time_s"] = df["time_us"] * 1.0e-6
    return df


def main() -> None:
    parameters = pd.read_csv(PARAMETER_PATH, encoding="utf-8-sig")
    df = read_data(VALIDATION_CSV)
    baseline_data = df.loc[df["time_s"] < PRE_IMPACT_END_S, "z_v"].to_numpy(dtype=float)
    baseline = float(np.mean(baseline_data))
    sigma = float(np.std(baseline_data))
    candidate = df[
        (df["time_s"] >= PRE_IMPACT_END_S)
        & (np.abs(df["z_v"] - baseline) >= DETECTION_SIGMA * sigma)
    ]
    impact_time = float(candidate.iloc[0]["time_s"])
    response = df[
        (df["time_s"] >= impact_time)
        & (df["time_s"] <= impact_time + ANALYSIS_DURATION_S)
    ].copy()
    time_s = response["time_s"].to_numpy(dtype=float) - impact_time
    measured = response["z_v"].to_numpy(dtype=float) - baseline

    predicted = np.full_like(time_s, IDENTIFIED_OFFSET_V)
    for row in parameters.itertuples(index=False):
        wn = 2.0 * np.pi * row.natural_frequency_hz
        alpha = row.damping_ratio * wn
        wd = wn * np.sqrt(1.0 - row.damping_ratio**2)
        predicted += np.exp(-alpha * time_s) * (
            row.cosine_coefficient_B * np.cos(wd * time_s)
            + row.sine_coefficient_C * np.sin(wd * time_s)
        )

    residual = measured - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    r_squared = float(1.0 - np.sum(residual**2) / np.sum((measured - np.mean(measured)) ** 2))

    table = pd.DataFrame(
        {
            "time_s": time_s,
            "validation_measured_z_v": measured,
            "prediction_from_run3_model_v": predicted,
            "residual_v": residual,
        }
    )
    table.to_csv(OUTPUT_DIR / "run4_validation_waveform.csv", index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    axes[0].plot(time_s * 1000.0, measured, color="#202020", lw=1.1, label="Run 4 measured")
    axes[0].plot(time_s * 1000.0, predicted, color="#D62728", lw=1.4, ls="--", label="Run 3 model, unchanged")
    axes[0].set_ylabel("Z output [V]")
    axes[0].set_title(f"Independent validation on run 4: R^2={r_squared:.3f}, RMSE={rmse:.3f} V")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(time_s * 1000.0, residual, color="#2166AC", lw=1.0)
    axes[1].axhline(0.0, color="#777777", lw=0.8, ls=":")
    axes[1].set_xlabel("Time from detected impact [ms]")
    axes[1].set_ylabel("Residual [V]")
    axes[1].set_title("Difference: measured - prediction")
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "run4_independent_validation.png", dpi=200)
    plt.close(figure)

    summary = (
        "Independent validation of the run-3 transfer function on 成功4回目.CSV\n"
        "No modal frequency, damping, amplitude, or phase was refitted.\n"
        f"Impact time: {impact_time:.7f} s\n"
        f"Measured peak amplitude: {np.max(np.abs(measured)):.9g} V\n"
        f"Predicted peak amplitude: {np.max(np.abs(predicted)):.9g} V\n"
        f"RMSE: {rmse:.9g} V\n"
        f"R^2: {r_squared:.9g}\n"
    )
    (OUTPUT_DIR / "run4_validation_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
