"""Refine the hammer-test transfer function by fitting the time waveform.

The matrix-pencil poles are retained, while each modal residue is fitted to
the measured ring-down.  This gives the real second-order modal form

    H_r(s) = (B_r*s + C0_r) /
             (s^2 + 2*alpha_r*s + wn_r^2)

whose unit-impulse response is

    h_r(t) = exp(-alpha_r*t) *
             (B_r*cos(wd_r*t) + C_r*sin(wd_r*t)).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import multi_mode_transfer_function_identification as identification


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "refined_time_domain_results"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = identification.read_gl980()
    impact_time = identification.detect_impact(df)
    baseline = float(
        df.loc[df["time_s"] < identification.PRE_IMPACT_END_S, "z_v"]
        .astype(float)
        .mean()
    )
    response = df[
        (df["time_s"] >= impact_time)
        & (df["time_s"] <= identification.ANALYSIS_END_S)
    ].copy()
    time_s = response["time_s"].to_numpy(dtype=float) - impact_time
    measured_raw = response["z_v"].to_numpy(dtype=float)
    measured = measured_raw - baseline
    dt = float(np.median(np.diff(time_s)))

    modes = identification.matrix_pencil_modes(measured, dt)
    modes = sorted(
        [mode for mode in modes if 600.0 <= mode["natural_frequency_hz"] <= 1250.0],
        key=lambda mode: mode["natural_frequency_hz"],
    )

    columns = []
    modal_constants = []
    for mode in modes:
        fn = mode["natural_frequency_hz"]
        zeta = mode["damping_ratio"]
        wn = 2.0 * np.pi * fn
        alpha = zeta * wn
        wd = wn * np.sqrt(1.0 - zeta**2)
        cosine = np.exp(-alpha * time_s) * np.cos(wd * time_s)
        sine = np.exp(-alpha * time_s) * np.sin(wd * time_s)
        columns.extend([cosine, sine])
        modal_constants.append((wn, alpha, wd))

    design = np.column_stack(columns + [np.ones_like(time_s)])
    coefficients, *_ = np.linalg.lstsq(design, measured, rcond=None)
    fitted_offset = float(coefficients[-1])
    fitted = design @ coefficients
    residual = measured - fitted
    rmse = float(np.sqrt(np.mean(residual**2)))
    r_squared = float(
        1.0 - np.sum(residual**2) / np.sum((measured - np.mean(measured)) ** 2)
    )

    modal_waveforms = []
    rows = []
    equation_lines = []
    for index, (mode, constants) in enumerate(zip(modes, modal_constants), start=1):
        wn, alpha, wd = constants
        cosine_coefficient = float(coefficients[2 * (index - 1)])
        sine_coefficient = float(coefficients[2 * (index - 1) + 1])
        waveform = np.exp(-alpha * time_s) * (
            cosine_coefficient * np.cos(wd * time_s)
            + sine_coefficient * np.sin(wd * time_s)
        )
        modal_waveforms.append(waveform)

        numerator_s = cosine_coefficient
        numerator_constant = cosine_coefficient * alpha + sine_coefficient * wd
        denominator_s = 2.0 * alpha
        denominator_constant = wn**2
        rows.append(
            {
                "mode": index,
                "natural_frequency_hz": mode["natural_frequency_hz"],
                "damping_ratio": mode["damping_ratio"],
                "damping_percent": 100.0 * mode["damping_ratio"],
                "decay_alpha_per_s": alpha,
                "cosine_coefficient_B": cosine_coefficient,
                "sine_coefficient_C": sine_coefficient,
                "numerator_s_coefficient": numerator_s,
                "numerator_constant": numerator_constant,
                "denominator_s_coefficient": denominator_s,
                "denominator_constant": denominator_constant,
            }
        )
        equation_lines.append(
            f"H_{index}(s) = ({numerator_s:.8g} s + {numerator_constant:.8g}) / "
            f"(s^2 + {denominator_s:.8g} s + {denominator_constant:.8g})"
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "refined_modal_parameters.csv", index=False, encoding="utf-8-sig")
    waveform_table = pd.DataFrame(
        {
            "time_s": time_s,
            "measured_raw_z_v": measured_raw,
            "measured_baseline_corrected_z_v": measured,
            "refined_model_z_v": fitted,
            "residual_v": residual,
        }
    )
    for index, waveform in enumerate(modal_waveforms, start=1):
        waveform_table[f"mode_{index}_v"] = waveform
    waveform_table.to_csv(OUTPUT_DIR / "refined_waveform.csv", index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(time_s * 1000.0, measured_raw, color="#202020", lw=1.1)
    axes[0].set_ylabel("Raw Z output [V]")
    axes[0].set_title("Original measured waveform")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s * 1000.0, measured, color="#202020", lw=1.1, label="Measured")
    axes[1].plot(
        time_s * 1000.0,
        fitted,
        color="#D62728",
        lw=1.4,
        linestyle="--",
        label=f"Refined transfer function (R^2={r_squared:.3f})",
    )
    axes[1].set_ylabel("Z output [V]")
    axes[1].set_title("Measured and reconstructed waveforms")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(time_s * 1000.0, residual, color="#2166AC", lw=1.0)
    axes[2].axhline(0.0, color="#777777", lw=0.8, linestyle=":")
    axes[2].set_xlabel("Time from detected impact [ms]")
    axes[2].set_ylabel("Residual [V]")
    axes[2].set_title(f"Difference: measured - model (RMSE={rmse:.3f} V)")
    axes[2].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "refined_waveform_comparison.png", dpi=200)
    plt.close(figure)

    summary = (
        "Time-domain refined transfer function from 成功3回目.CSV\n"
        f"Number of modes: {len(modes)}\n"
        f"Fitted residual offset: {fitted_offset:.9g} V\n"
        f"RMSE: {rmse:.9g} V\n"
        f"R^2: {r_squared:.9g}\n\n"
        + table.to_string(index=False)
        + "\n\nH_ZZ(s) = "
        + " + ".join(f"H_{index}(s)" for index in range(1, len(modes) + 1))
        + "\n"
        + "\n".join(equation_lines)
        + "\n"
    )
    (OUTPUT_DIR / "refined_transfer_function.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
