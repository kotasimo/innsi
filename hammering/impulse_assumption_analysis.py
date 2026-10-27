"""Estimate relative transfer functions under a unit-impulse assumption.

Assumption:
    The hammer input is a unit impulse at the beginning of the selected window.
    Therefore F(f) = 1 and H_i(f) = Y_i(f).

The result is normalized by the largest magnitude across X/Y/Z because the
actual impulse magnitude was not recorded.  It is therefore a relative
amplitude transfer function, not an absolute calibrated FRF.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks, windows


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "成功3回目.CSV"
OUTPUT_DIR = ROOT / "impulse_assumption_results"
START_S = 0.0286
END_S = 0.0878
FREQ_MIN_HZ = 100.0
FREQ_MAX_HZ = 1300.0


def read_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, skiprows=20, header=None, encoding="utf-8-sig")
    df = df.iloc[:, :9]
    df.columns = [
        "number", "date", "clock_time", "time_us", "x_v", "y_v", "z_v",
        "alarm", "alarm_out",
    ]
    for column in ["time_us", "x_v", "y_v", "z_v"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["time_us", "x_v", "y_v", "z_v"]).reset_index(drop=True)
    df["time_s"] = df["time_us"] * 1.0e-6
    return df


def complex_spectrum(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    values = detrend(np.asarray(values, dtype=float), type="constant")
    window = windows.hann(len(values), sym=False)
    spectrum = np.fft.rfft(values * window) / np.sum(window)
    frequency = np.fft.rfftfreq(len(values), d=1.0 / fs)
    return frequency, spectrum


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = read_data()
    dt = float(np.median(np.diff(df["time_s"].to_numpy())))
    fs = 1.0 / dt
    section = df[(df["time_s"] >= START_S) & (df["time_s"] <= END_S)].copy()

    spectra: dict[str, np.ndarray] = {}
    frequency = None
    for axis, column in {"X": "x_v", "Y": "y_v", "Z": "z_v"}.items():
        frequency, spectrum = complex_spectrum(section[column].to_numpy(), fs)
        spectra[axis] = spectrum

    assert frequency is not None
    band = (frequency >= FREQ_MIN_HZ) & (frequency <= FREQ_MAX_HZ)
    reference = max(float(np.max(np.abs(spectra[axis][band]))) for axis in spectra)

    rows = []
    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = {"X": "#2474B5", "Y": "#0A7A78", "Z": "#E57A2B"}

    for axis in ["X", "Y", "Z"]:
        magnitude = np.abs(spectra[axis]) / reference
        ax.plot(frequency[band], magnitude[band], label=f"{axis}-axis", color=colors[axis], linewidth=1.6)

        band_mag = magnitude[band]
        band_freq = frequency[band]
        peak_indices, _ = find_peaks(
            band_mag,
            prominence=max(float(np.max(band_mag)) * 0.03, np.finfo(float).eps),
        )
        ranked = sorted(peak_indices, key=lambda i: band_mag[i], reverse=True)[:5]
        for i in ranked:
            rows.append({
                "axis": axis,
                "frequency_hz": float(band_freq[i]),
                "relative_magnitude": float(band_mag[i]),
            })

    ax.set_title("Approximate transfer-function magnitude under unit-impulse assumption")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Relative magnitude [-]")
    ax.set_xlim(FREQ_MIN_HZ, FREQ_MAX_HZ)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "relative_transfer_functions_xyz.png", dpi=200)
    plt.close(fig)

    peaks = pd.DataFrame(rows).sort_values(["axis", "relative_magnitude"], ascending=[True, False])
    peaks.to_csv(OUTPUT_DIR / "relative_transfer_function_peaks.csv", index=False, encoding="utf-8-sig")

    result = pd.DataFrame({"frequency_hz": frequency[band]})
    for axis in ["X", "Y", "Z"]:
        result[f"H_{axis}_relative_magnitude"] = np.abs(spectra[axis][band]) / reference
        result[f"H_{axis}_phase_deg"] = np.angle(spectra[axis][band], deg=True)
    result.to_csv(OUTPUT_DIR / "relative_transfer_functions_xyz.csv", index=False, encoding="utf-8-sig")

    phase_fig, phase_ax = plt.subplots(figsize=(12, 6.5))
    for axis in ["X", "Y", "Z"]:
        phase_ax.plot(
            frequency[band],
            np.angle(spectra[axis][band], deg=True),
            label=f"{axis}-axis",
            color=colors[axis],
            linewidth=1.2,
        )
    phase_ax.set_title("Approximate transfer-function phase under unit-impulse assumption")
    phase_ax.set_xlabel("Frequency [Hz]")
    phase_ax.set_ylabel("Phase [deg]")
    phase_ax.set_xlim(FREQ_MIN_HZ, FREQ_MAX_HZ)
    phase_ax.set_ylim(-180, 180)
    phase_ax.grid(True, alpha=0.35)
    phase_ax.legend()
    phase_fig.tight_layout()
    phase_fig.savefig(OUTPUT_DIR / "relative_transfer_function_phase_xyz.png", dpi=200)
    plt.close(phase_fig)

    print(f"sampling_frequency_hz={fs:.3f}")
    print(f"frequency_resolution_hz={fs / len(section):.3f}")
    print(peaks.to_string(index=False))
    print(f"output_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
