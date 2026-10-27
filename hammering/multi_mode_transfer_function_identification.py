"""Estimate a four-mode H_ZZ(s) from the ordinary-hammer ring-down.

The Z-direction hammer strike is assumed to be a unit impulse. Natural
frequencies and damping ratios are obtained from the complex poles of the
ring-down by a matrix-pencil calculation. The gain at each peak is read from
the densely sampled FFT and converted to a canonical second-order modal gain.
"""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "成功3回目.CSV"
OUTPUT_DIR = ROOT / "multi_mode_identification_results"
PRE_IMPACT_END_S = 0.025
ANALYSIS_END_S = 0.0878
DETECTION_SIGMA = 8.0
MATRIX_PENCIL_RANK = 20
FFT_LENGTH = 131072
PEAK_WINDOWS_HZ = np.array([[660, 740], [790, 870], [890, 950], [1090, 1170]], dtype=float)


def read_gl980() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, skiprows=20, header=None, encoding="utf-8-sig")
    df = df.iloc[:, :9].copy()
    df.columns = ["number", "date", "clock_time", "time_us", "x_v", "y_v", "z_v", "alarm", "alarm_out"]
    for name in ["time_us", "x_v", "y_v", "z_v"]:
        df.loc[:, name] = pd.to_numeric(df[name], errors="coerce")
    df = df.dropna(subset=["time_us", "x_v", "y_v", "z_v"]).reset_index(drop=True)
    df["time_s"] = df["time_us"].astype(float) * 1.0e-6
    return df


def detect_impact(df: pd.DataFrame) -> float:
    baseline = df.loc[df["time_s"] < PRE_IMPACT_END_S, "z_v"].to_numpy(dtype=float)
    mean, sigma = float(np.mean(baseline)), float(np.std(baseline))
    candidate = df.loc[
        (df["time_s"] >= PRE_IMPACT_END_S)
        & (np.abs(df["z_v"].astype(float) - mean) >= DETECTION_SIGMA * sigma)
    ]
    if candidate.empty:
        raise RuntimeError("Impact could not be detected.")
    return float(candidate.iloc[0]["time_s"])


def matrix_pencil_modes(signal: np.ndarray, dt: float) -> list[dict[str, float]]:
    sample_count = len(signal)
    row_count = sample_count // 2
    column_count = sample_count - row_count
    y0 = np.column_stack([signal[j : j + row_count] for j in range(column_count)])
    y1 = np.column_stack([signal[j + 1 : j + row_count + 1] for j in range(column_count)])
    u, singular_values, vh = np.linalg.svd(y0, full_matrices=False)
    rank = min(MATRIX_PENCIL_RANK, len(singular_values))
    reduced = (
        np.diag(1.0 / singular_values[:rank])
        @ u[:, :rank].conj().T
        @ y1
        @ vh[:rank, :].conj().T
    )
    continuous_poles = np.log(np.linalg.eigvals(reduced)) / dt
    modes = []
    for pole in continuous_poles:
        if pole.imag <= 0 or pole.real >= 0:
            continue
        natural_omega = abs(pole)
        natural_frequency = natural_omega / (2 * np.pi)
        damping_ratio = -pole.real / natural_omega
        if 100 <= natural_frequency <= 1300 and 0 < damping_ratio < 0.20:
            modes.append({
                "natural_frequency_hz": float(natural_frequency),
                "damped_frequency_hz": float(pole.imag / (2 * np.pi)),
                "damping_ratio": float(damping_ratio),
            })
    return modes


def modal_frequency_response(frequency_hz, natural_frequency_hz, damping_ratio, signed_gain):
    omega = 2 * np.pi * frequency_hz
    wn = 2 * np.pi * natural_frequency_hz
    return signed_gain * wn**2 / (-omega**2 + 1j * 2 * damping_ratio * wn * omega + wn**2)


def choose_signs_and_time_shift(frequency, measured_h, natural_frequencies, damping_ratios, gains):
    band = (frequency >= 600) & (frequency <= 1250)
    f, measured = frequency[band], measured_h[band]
    basis = np.column_stack([
        modal_frequency_response(f, fn, zeta, 1.0)
        for fn, zeta in zip(natural_frequencies, damping_ratios)
    ])
    omega = 2 * np.pi * f
    best = (np.inf, None, 0.0)
    for signs in product([-1.0, 1.0], repeat=len(gains)):
        base = basis @ (gains * np.asarray(signs))
        for shift in np.linspace(-1.0e-3, 1.0e-3, 2001):
            error = float(np.mean(np.abs(np.exp(1j * omega * shift) * base - measured) ** 2))
            if error < best[0]:
                best = (error, np.asarray(signs), float(shift))
    if best[1] is None:
        raise RuntimeError("Modal sign fitting failed.")
    return best[1], best[2]


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = read_gl980()
    impact_time = detect_impact(df)
    baseline = float(df.loc[df["time_s"] < PRE_IMPACT_END_S, "z_v"].astype(float).mean())
    response = df[(df["time_s"] >= impact_time) & (df["time_s"] <= ANALYSIS_END_S)].copy()
    t = response["time_s"].to_numpy(dtype=float) - impact_time
    measured = response["z_v"].to_numpy(dtype=float) - baseline
    dt = float(np.median(np.diff(t)))

    frequency = np.fft.rfftfreq(FFT_LENGTH, dt)
    measured_h = dt * np.fft.rfft(measured, FFT_LENGTH)
    magnitude = np.abs(measured_h)
    peak_indices, _ = find_peaks(magnitude)
    selected_fft_peaks = []
    for lower, upper in PEAK_WINDOWS_HZ:
        candidates = peak_indices[(frequency[peak_indices] >= lower) & (frequency[peak_indices] <= upper)]
        if len(candidates) == 0:
            raise RuntimeError(f"No FFT peak found in {lower:.0f}-{upper:.0f} Hz.")
        selected_fft_peaks.append(int(candidates[np.argmax(magnitude[candidates])]))

    unused_modes = matrix_pencil_modes(measured, dt)
    selected_modes = []
    for peak_index in selected_fft_peaks:
        peak_frequency = float(frequency[peak_index])
        mode = min(unused_modes, key=lambda item: abs(item["damped_frequency_hz"] - peak_frequency))
        if abs(mode["damped_frequency_hz"] - peak_frequency) > 55:
            raise RuntimeError(f"No ring-down pole found near {peak_frequency:.1f} Hz.")
        selected_modes.append(mode)
        unused_modes.remove(mode)

    natural_frequencies = np.asarray([m["natural_frequency_hz"] for m in selected_modes])
    damping_ratios = np.asarray([m["damping_ratio"] for m in selected_modes])
    peak_frequencies = frequency[selected_fft_peaks]
    peak_gains = magnitude[selected_fft_peaks]
    gain_magnitudes = 2 * damping_ratios * np.sqrt(1 - damping_ratios**2) * peak_gains
    signs, phase_time_shift = choose_signs_and_time_shift(
        frequency, measured_h, natural_frequencies, damping_ratios, gain_magnitudes
    )
    signed_gains = signs * gain_magnitudes

    rows, modal_h_values = [], []
    for index, (mode, peak_frequency, peak_gain, gain_magnitude, signed_gain) in enumerate(
        zip(selected_modes, peak_frequencies, peak_gains, gain_magnitudes, signed_gains), start=1
    ):
        fn, zeta = mode["natural_frequency_hz"], mode["damping_ratio"]
        wn = 2 * np.pi * fn
        modal_h_values.append(modal_frequency_response(frequency, fn, zeta, signed_gain))
        rows.append({
            "mode": index,
            "fft_peak_frequency_hz": peak_frequency,
            "natural_frequency_hz": fn,
            "damping_ratio": zeta,
            "damping_percent": 100 * zeta,
            "measured_peak_gain": peak_gain,
            "model_gain_magnitude_K": gain_magnitude,
            "signed_model_gain_K": signed_gain,
            "numerator_constant": signed_gain * wn**2,
            "denominator_s_coefficient": 2 * zeta * wn,
            "denominator_constant": wn**2,
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "identified_modal_parameters.csv", index=False, encoding="utf-8-sig")

    total_h = np.sum(np.asarray(modal_h_values), axis=0)
    shifted_total_h = np.exp(1j * 2 * np.pi * frequency * phase_time_shift) * total_h
    plot_band = (frequency >= 100) & (frequency <= 1300)
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(frequency[plot_band], magnitude[plot_band], color="#202020", lw=1.5, label="Measured H_ZZ")
    colors = ["#2166AC", "#1B9E77", "#D95F02", "#7570B3"]
    for index, modal_h in enumerate(modal_h_values):
        axes[0].plot(frequency[plot_band], np.abs(modal_h[plot_band]), color=colors[index], lw=1.0, label=f"Mode {index + 1}")
    axes[0].plot(frequency[plot_band], np.abs(shifted_total_h[plot_band]), color="#D62728", lw=1.5, ls="--", label="Four-mode model")
    axes[0].set_ylabel("Gain |H_ZZ| [V/N, assumed]")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3)
    axes[1].plot(frequency[plot_band], np.unwrap(np.angle(measured_h[plot_band])) * 180 / np.pi, color="#202020", lw=1.2, label="Measured")
    axes[1].plot(frequency[plot_band], np.unwrap(np.angle(shifted_total_h[plot_band])) * 180 / np.pi, color="#D62728", lw=1.4, ls="--", label="Model")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Unwrapped phase [deg]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.suptitle("Four-mode transfer-function identification from the Z ring-down")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "identified_transfer_function.png", dpi=200)
    plt.close(figure)

    response_figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.plot(t, measured, color="#202020", lw=1.1)
    axis.set_xlabel("Time from detected impact [s]")
    axis.set_ylabel("Z output [V]")
    axis.set_title("Ring-down used for pole and damping identification")
    axis.grid(True, alpha=0.3)
    response_figure.tight_layout()
    response_figure.savefig(OUTPUT_DIR / "ringdown_used_for_identification.png", dpi=200)
    plt.close(response_figure)

    equation_lines = [
        f"H_{int(row['mode'])}(s) = ({row['numerator_constant']:.8g}) / "
        f"(s^2 + {row['denominator_s_coefficient']:.8g} s + {row['denominator_constant']:.8g})"
        for row in rows
    ]
    summary = (
        "Four-mode H_ZZ(s) from 成功3回目.CSV\n"
        "Assumption: the Z-direction hammer strike is a unit impulse.\n"
        f"Detected impact time: {impact_time:.7f} s\n"
        f"Sampling frequency: {1 / dt:.3f} Hz\n"
        f"Matrix-pencil rank: {MATRIX_PENCIL_RANK}\n"
        f"Common fitted phase-equivalent time shift: {phase_time_shift:.9f} s\n\n"
        + table.to_string(index=False)
        + "\n\nH_ZZ(s) = H_1(s) + H_2(s) + H_3(s) + H_4(s)\n"
        + "\n".join(equation_lines)
        + "\n"
    )
    (OUTPUT_DIR / "identified_transfer_function.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
