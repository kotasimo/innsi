"""Analyze two GL980 records using the laboratory procedure PDF.

The script keeps the source CSV unchanged and writes figures/tables beside each
record.  The selected signal is CH3 because the existing hammering workflow
uses the third (Z/load-cell) channel; this choice is recorded in metadata.txt.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "共和" / "2026-10-14" / "kyowa50N_成功1.CSV",
    ROOT / "日章" / "10000HZ.csv",
]
FMAX_HZ = (500.0, 1000.0, 1500.0, 2000.0, 3000.0)


def read_gl980(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    raw = pd.read_csv(path, skiprows=20, header=None, encoding="utf-8-sig").iloc[:, :9]
    raw.columns = ["number", "date", "clock", "time_us", "ch1", "ch2", "ch3", "alarm", "alarm_out"]
    for column in ("time_us", "ch1", "ch2", "ch3"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.dropna(subset=["time_us", "ch3"]).reset_index(drop=True)
    metadata = {
        "sampling_interval_us": str(float(np.median(np.diff(raw["time_us"].to_numpy())))),
        "points": str(len(raw)),
        "channel": "CH3",
    }
    return raw["time_us"].to_numpy(float) * 1e-6, raw["ch3"].to_numpy(float), metadata


def find_hit(time: np.ndarray, output: np.ndarray) -> tuple[int, float, float, float]:
    baseline_end = min(0.02, time[-1] * 0.1)
    baseline = output[time <= baseline_end]
    mean = float(np.mean(baseline))
    sigma = float(np.std(baseline, ddof=1))
    threshold = max(6.0 * sigma, 1e-9)
    candidates = np.flatnonzero((time >= baseline_end) & (np.abs(output - mean) >= threshold))
    if len(candidates) == 0:
        candidates = np.array([int(np.argmax(np.abs(output - mean)))])
    hit = int(candidates[0])
    return hit, mean, sigma, threshold


def fft_peaks(time: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = float(np.median(np.diff(time)))
    frequency = np.fft.rfftfreq(len(values), dt)
    spectrum = np.fft.rfft(values)
    amplitude = np.abs(spectrum) * 2.0 / len(values)
    amplitude[[0, -1]] *= 0.5
    peaks, properties = find_peaks(amplitude, distance=2, prominence=max(np.max(amplitude) * 0.02, 1e-12))
    peaks = peaks[(frequency[peaks] >= 20.0) & (frequency[peaks] <= frequency[-1] * 0.9)]
    if len(peaks) == 0:
        raise RuntimeError("FFT peak was not found.")
    selected = peaks[np.argsort(amplitude[peaks])[::-1][:8]]
    selected = selected[np.argsort(frequency[selected])]
    return frequency, amplitude, selected


def fit_modes(time: np.ndarray, values: np.ndarray, peak_frequency: np.ndarray) -> tuple[np.ndarray, float]:
    # Keep the model size data-driven: retain prominent FFT candidates, then
    # fit frequency, damping, sine/cosine residue, and a constant offset.
    mode_count = min(len(peak_frequency), 8)
    frequencies = peak_frequency[:mode_count]
    t = time - time[0]
    scale = max(float(np.std(values)), 1e-9)
    columns = []
    for frequency in frequencies:
        columns.extend([np.cos(2 * np.pi * frequency * t), np.sin(2 * np.pi * frequency * t)])
    design = np.column_stack(columns + [np.ones_like(t)])
    initial_linear, *_ = np.linalg.lstsq(design, values, rcond=None)
    x0 = []
    lower = []
    upper = []
    for index, frequency in enumerate(frequencies):
        x0.extend([frequency, 0.01, initial_linear[2 * index], initial_linear[2 * index + 1]])
        lower.extend([max(5.0, frequency - 30.0), 1e-5, -20 * scale, -20 * scale])
        upper.extend([frequency + 30.0, 0.35, 20 * scale, 20 * scale])
    x0.append(float(initial_linear[-1]))
    lower.append(float(np.min(values) - 5 * scale))
    upper.append(float(np.max(values) + 5 * scale))

    def model(parameters: np.ndarray) -> np.ndarray:
        result = np.full_like(t, parameters[-1], dtype=float)
        for index in range(mode_count):
            frequency, zeta, cosine, sine = parameters[4 * index : 4 * index + 4]
            wn = 2 * np.pi * frequency
            alpha = zeta * wn
            wd = wn * np.sqrt(max(1e-12, 1 - zeta * zeta))
            result += np.exp(-alpha * t) * (cosine * np.cos(wd * t) + sine * np.sin(wd * t))
        return result

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (model(parameters) - values) / scale

    fitted = least_squares(residual, x0, bounds=(lower, upper), max_nfev=1500, x_scale="jac")
    parameters = fitted.x
    rows = []
    for index in range(mode_count):
        frequency, zeta, cosine, sine = parameters[4 * index : 4 * index + 4]
        wn = 2 * np.pi * frequency
        wd = wn * np.sqrt(max(1e-12, 1 - zeta * zeta))
        amplitude = float(np.hypot(cosine, sine))
        # Unit-impulse canonical K is reported using the fitted amplitude;
        # the phase/residue mismatch is retained in the metadata warning.
        gain = amplitude * wd / (wn * wn)
        rows.append((index + 1, frequency, zeta, wn, wd, cosine, sine, amplitude, gain))
    fitted_values = model(parameters)
    rmse = float(np.sqrt(np.mean((fitted_values - values) ** 2)))
    return np.asarray(rows), rmse


def transfer_response(frequency: np.ndarray, modes: np.ndarray) -> np.ndarray:
    omega = 2 * np.pi * frequency
    response = np.zeros_like(omega, dtype=complex)
    for row in modes:
        _, _, zeta, wn, _, _, _, _, gain = row
        response += gain * wn * wn / (-omega * omega + 1j * 2 * zeta * wn * omega + wn * wn)
    return response


def analyze(path: Path) -> Path:
    time, output, metadata = read_gl980(path)
    hit_index, baseline, sigma, threshold = find_hit(time, output)
    hit_time = float(time[hit_index])
    response_time = time[hit_index:] - hit_time
    response_output = output[hit_index:]
    frequency, amplitude, peaks = fft_peaks(response_time, response_output)
    modes, rmse = fit_modes(response_time, response_output, frequency[peaks])
    # Keep results directly below the laboratory/vendor folder.  If the input
    # is inside a dated measurement folder, add the source stem to avoid
    # collisions with future records.
    if path.parent.name.startswith("2026-"):
        output_dir = path.parent.parent / "解析結果" / path.stem
    else:
        output_dir = path.parent / "解析結果" / path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    full_frequency, full_amplitude, _ = fft_peaks(time, output)
    pd.DataFrame({"frequency_hz": full_frequency, "amplitude_v": full_amplitude}).to_csv(
        output_dir / "fft_spectrum.csv", index=False, encoding="utf-8-sig"
    )
    columns = ["mode", "natural_frequency_hz", "damping_ratio", "natural_omega_rad_s", "damped_omega_rad_s", "cosine_coefficient_v", "sine_coefficient_v", "fitted_amplitude_v", "K_relative"]
    pd.DataFrame(modes, columns=columns).to_csv(output_dir / "identified_modes.csv", index=False, encoding="utf-8-sig")

    # FFT coefficients of the selected output interval and inverse transfer
    # function at five valid maximum frequencies.
    dt = float(np.median(np.diff(response_time)))
    response_frequency = np.fft.rfftfreq(len(response_output), dt)
    output_hat = np.fft.rfft(response_output)
    G = transfer_response(response_frequency, modes)
    input_columns: dict[str, np.ndarray] = {"time_s": response_time}
    stats = []
    for fmax in FMAX_HZ:
        mask = response_frequency <= min(fmax, response_frequency[-1])
        input_hat = np.zeros_like(output_hat)
        safe = mask & (np.abs(G) > 1e-12)
        input_hat[safe] = output_hat[safe] / G[safe]
        estimated = np.fft.irfft(input_hat, n=len(response_output)).real
        label = f"input_fmax_{int(fmax)}hz"
        input_columns[label] = estimated
        peak_index = int(np.argmax(np.abs(estimated)))
        stats.append((fmax, float(np.max(estimated)), float(np.min(estimated)), float(response_time[peak_index]), float(np.max(np.abs(estimated)))))
    pd.DataFrame(input_columns).to_csv(output_dir / "estimated_inputs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stats, columns=["fmax_hz", "maximum", "minimum", "peak_time_s", "absolute_peak"]).to_csv(output_dir / "estimated_input_statistics.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(time, output, lw=0.8)
    ax.axvline(hit_time, color="tab:red", ls="--", label=f"hit={hit_time:.6f} s")
    ax.set(xlabel="Time [s]", ylabel="Output [V]", title="Measured decaying output (CH3)")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "01_output_waveform.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(full_frequency, full_amplitude, lw=0.8)
    for row in modes:
        ax.axvline(row[1], color="tab:red", alpha=0.5)
    ax.set(xlim=(0, full_frequency[-1]), xlabel="Frequency [Hz]", ylabel="Amplitude [V]", title="FFT amplitude spectrum")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(output_dir / "02_fft_spectrum.png", dpi=180); plt.close(fig)

    estimated = pd.DataFrame(input_columns)
    fig, axes = plt.subplots(len(FMAX_HZ), 1, figsize=(11, 11), sharex=True)
    for axis, fmax in zip(np.atleast_1d(axes), FMAX_HZ):
        axis.plot(response_time, estimated[f"input_fmax_{int(fmax)}hz"], lw=0.8)
        axis.set_ylabel(f"{int(fmax)} Hz")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time [s]"); fig.suptitle("Estimated periodic input particular solution"); fig.tight_layout(); fig.savefig(output_dir / "03_estimated_inputs.png", dpi=180); plt.close(fig)

    metadata_lines = [
        f"source={path}", "procedure_source=研究方針/解析手順２.pdf", "channel=CH3", "input_units=relative unit (not calibrated N)",
        f"sampling_interval_s={metadata['sampling_interval_us']}e-6", f"sampling_frequency_hz={1 / dt:.9g}", f"points={metadata['points']}",
        f"hit_time_s={hit_time:.9g}", f"baseline_mean_v={baseline:.9g}", f"baseline_std_v={sigma:.9g}", f"hit_threshold_v={threshold:.9g}",
        f"response_interval_s={response_time[0]:.9g}..{response_time[-1]:.9g}", f"mode_count={len(modes)}", f"fit_rmse_v={rmse:.9g}",
        "transfer_model=G(s)=sum K_r*omega_r^2/(s^2+2*zeta_r*omega_r*s+omega_r^2)",
        "gain_note=K_relative uses fitted damped-response amplitude under unit-impulse assumption; arbitrary phase/residue and unknown hammer force limit absolute gain.",
        "input_method=FFT coefficients of selected output interval divided by G(j*omega), equivalent to the periodic particular solution coefficient calculation.",
        "fmax_hz=" + ",".join(str(int(value)) for value in FMAX_HZ),
    ]
    (output_dir / "metadata.txt").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    with (output_dir / "transfer_function.txt").open("w", encoding="utf-8") as handle:
        handle.write("G(s) = sum_r K_r*omega_r^2/(s^2 + 2*zeta_r*omega_r*s + omega_r^2)\n\n")
        for row in modes:
            handle.write(f"mode {int(row[0])}: fn={row[1]:.9g} Hz, zeta={row[2]:.9g}, omega={row[3]:.9g} rad/s, K_relative={row[8]:.9g}\n")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_INPUTS)
    args = parser.parse_args()
    for path in args.paths:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Analyzing {path}")
        print(f"Results: {analyze(path)}")


if __name__ == "__main__":
    main()
