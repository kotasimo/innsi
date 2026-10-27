"""Reconstruct the Z-output waveform from the identified modal H_ZZ(s).

For a unit-impulse input, the output is the impulse response.  Each identified
second-order term

    H_r(s) = K_r*wn_r^2 / (s^2 + 2*zeta_r*wn_r*s + wn_r^2)

has the time response

    h_r(t) = K_r*wn_r^2/wd_r * exp(-zeta_r*wn_r*t) * sin(wd_r*t).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import multi_mode_transfer_function_identification as identification


ROOT = Path(__file__).resolve().parent
PARAMETER_PATH = ROOT / "multi_mode_identification_results" / "identified_modal_parameters.csv"
OUTPUT_DIR = ROOT / "multi_mode_identification_results"
OUTPUT_CSV = OUTPUT_DIR / "reconstructed_output_waveform.csv"
OUTPUT_PNG = OUTPUT_DIR / "reconstructed_output_waveform.png"
PHASE_EQUIVALENT_DELAY_S = 0.000444


def modal_impulse_response(time_s, natural_frequency_hz, damping_ratio, signed_gain):
    wn = 2.0 * np.pi * natural_frequency_hz
    wd = wn * np.sqrt(1.0 - damping_ratio**2)
    return (
        signed_gain
        * wn**2
        / wd
        * np.exp(-damping_ratio * wn * time_s)
        * np.sin(wd * time_s)
    )


def delayed(values: np.ndarray, delay_samples: int) -> np.ndarray:
    result = np.zeros_like(values)
    if delay_samples == 0:
        return values.copy()
    if delay_samples > 0:
        result[delay_samples:] = values[:-delay_samples]
    else:
        result[:delay_samples] = values[-delay_samples:]
    return result


def main() -> None:
    parameters = pd.read_csv(PARAMETER_PATH, encoding="utf-8-sig")

    df = identification.read_gl980()
    impact_time = identification.detect_impact(df)
    baseline = float(
        df.loc[df["time_s"] < identification.PRE_IMPACT_END_S, "z_v"]
        .astype(float)
        .mean()
    )
    measured_frame = df[
        (df["time_s"] >= impact_time)
        & (df["time_s"] <= identification.ANALYSIS_END_S)
    ].copy()
    time_s = measured_frame["time_s"].to_numpy(dtype=float) - impact_time
    measured_raw = measured_frame["z_v"].to_numpy(dtype=float)
    measured = measured_raw - baseline
    dt = float(np.median(np.diff(time_s)))

    modal_values = []
    for row in parameters.itertuples(index=False):
        modal_values.append(
            modal_impulse_response(
                time_s,
                row.natural_frequency_hz,
                row.damping_ratio,
                row.signed_model_gain_K,
            )
        )
    modal_values = np.asarray(modal_values)
    reconstructed = np.sum(modal_values, axis=0)
    delay_samples = int(round(PHASE_EQUIVALENT_DELAY_S / dt))
    reconstructed_delayed = delayed(reconstructed, delay_samples)

    output = pd.DataFrame(
        {
            "time_s": time_s,
            "measured_raw_z_v": measured_raw,
            "measured_z_v": measured,
            "mode_1_v": modal_values[0],
            "mode_2_v": modal_values[1],
            "mode_3_v": modal_values[2],
            "mode_4_v": modal_values[3],
            "reconstructed_z_v": reconstructed,
            "reconstructed_with_identified_delay_v": reconstructed_delayed,
        }
    )
    output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    colors = ["#2166AC", "#1B9E77", "#D95F02", "#7570B3"]
    figure, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(
        time_s * 1000.0,
        measured_raw,
        color="#202020",
        linewidth=1.1,
        label="Raw measured Z output",
    )
    axes[0].axhline(
        baseline,
        color="#7F7F7F",
        linewidth=1.0,
        linestyle=":",
        label=f"Pre-impact mean: {baseline:.3f} V",
    )
    axes[0].set_ylabel("Raw Z output [V]")
    axes[0].set_title("Original measured waveform without baseline subtraction")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for index, mode_values in enumerate(modal_values):
        axes[1].plot(
            time_s * 1000.0,
            mode_values,
            color=colors[index],
            linewidth=1.0,
            label=f"Mode {index + 1}",
        )
    axes[1].plot(
        time_s * 1000.0,
        reconstructed,
        color="#D62728",
        linewidth=1.8,
        label="Sum from H_ZZ(s)",
    )
    axes[1].set_ylabel("Calculated output [V]")
    axes[1].set_title("Impulse-output components calculated from the transfer function")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=3)

    axes[2].plot(
        time_s * 1000.0,
        measured,
        color="#202020",
        linewidth=1.1,
        label="Measured hammer response",
    )
    axes[2].plot(
        time_s * 1000.0,
        reconstructed_delayed,
        color="#D62728",
        linewidth=1.4,
        linestyle="--",
        label=f"Calculated response (delay {delay_samples * dt * 1000:.1f} ms)",
    )
    axes[2].set_xlabel("Time from detected impact [ms]")
    axes[2].set_ylabel("Z output [V]")
    axes[2].set_title("Baseline-corrected measurement and reconstructed output")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=200)
    plt.close(figure)

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Sampling interval: {dt:.7f} s")
    print(f"Applied comparison delay: {delay_samples * dt:.7f} s")
    print(f"Calculated peak amplitude: {np.max(np.abs(reconstructed_delayed)):.6f} V")
    print(f"Measured peak amplitude: {np.max(np.abs(measured)):.6f} V")


if __name__ == "__main__":
    main()
