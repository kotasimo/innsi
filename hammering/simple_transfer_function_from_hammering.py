"""Simplified transfer function from the ordinary-hammer test.

The impact is assumed to be a unit impulse applied in the Z direction:

    f_z(t) = delta(t),  F_z(f) = 1

Therefore the three transfer paths are estimated directly from the complex
Fourier transforms of the measured load-cell voltage responses:

    H_XZ(f) = V_X(f)
    H_YZ(f) = V_Y(f)
    H_ZZ(f) = V_Z(f)

The impulse is detected from the Z channel.  A pre-impact mean is subtracted,
the response is cut from the detected impact to the established end time, and
the continuous-time Fourier transform is approximated by dt * rFFT.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "成功3回目.CSV"
OUTPUT_DIR = ROOT / "simple_transfer_function_results"
PRE_IMPACT_END_S = 0.025
ANALYSIS_END_S = 0.0878
FREQ_MIN_HZ = 100.0
FREQ_MAX_HZ = 1300.0
DETECTION_SIGMA = 8.0


def read_gl980() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, skiprows=20, header=None, encoding="utf-8-sig")
    df = df.iloc[:, :9].copy()
    df.columns = [
        "number", "date", "clock_time", "time_us", "x_v", "y_v", "z_v",
        "alarm", "alarm_out",
    ]
    for name in ["time_us", "x_v", "y_v", "z_v"]:
        df.loc[:, name] = pd.to_numeric(df[name], errors="coerce")
    df = df.dropna(subset=["time_us", "x_v", "y_v", "z_v"]).reset_index(drop=True)
    df["time_s"] = df["time_us"].astype(float) * 1.0e-6
    return df


def detect_impact(df: pd.DataFrame) -> float:
    baseline = df.loc[df["time_s"] < PRE_IMPACT_END_S, "z_v"].to_numpy(dtype=float)
    mean = float(np.mean(baseline))
    sigma = float(np.std(baseline))
    candidate = df.loc[
        (df["time_s"] >= PRE_IMPACT_END_S)
        & (np.abs(df["z_v"].astype(float) - mean) >= DETECTION_SIGMA * sigma)
    ]
    if candidate.empty:
        raise RuntimeError("Impact could not be detected.")
    return float(candidate.iloc[0]["time_s"])


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = read_gl980()
    time = df["time_s"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(time)))
    fs = 1.0 / dt
    impact_time = detect_impact(df)

    response = df[(df["time_s"] >= impact_time) & (df["time_s"] <= ANALYSIS_END_S)].copy()
    relative_time = response["time_s"].to_numpy(dtype=float) - impact_time
    frequency = np.fft.rfftfreq(len(response), d=dt)
    band = (frequency >= FREQ_MIN_HZ) & (frequency <= FREQ_MAX_HZ)

    baseline_mask = df["time_s"] < PRE_IMPACT_END_S
    channel_map = {"XZ": "x_v", "YZ": "y_v", "ZZ": "z_v"}
    colors = {"XZ": "#2474B5", "YZ": "#0A7A78", "ZZ": "#E57A2B"}
    transfer: dict[str, np.ndarray] = {}

    for path_name, column in channel_map.items():
        baseline_mean = float(df.loc[baseline_mask, column].astype(float).mean())
        values = response[column].to_numpy(dtype=float) - baseline_mean
        transfer[path_name] = dt * np.fft.rfft(values)

    table = pd.DataFrame({"frequency_hz": frequency[band]})
    for path_name in ["XZ", "YZ", "ZZ"]:
        table[f"H_{path_name}_real"] = transfer[path_name][band].real
        table[f"H_{path_name}_imag"] = transfer[path_name][band].imag
        table[f"H_{path_name}_magnitude"] = np.abs(transfer[path_name][band])
        table[f"H_{path_name}_phase_deg"] = np.angle(transfer[path_name][band], deg=True)
    table.to_csv(OUTPUT_DIR / "simple_transfer_functions_complex.csv", index=False, encoding="utf-8-sig")

    peaks_rows = []
    magnitude_fig, magnitude_ax = plt.subplots(figsize=(12, 6.5))
    for path_name in ["XZ", "YZ", "ZZ"]:
        band_frequency = frequency[band]
        magnitude = np.abs(transfer[path_name][band])
        magnitude_ax.plot(
            band_frequency,
            magnitude,
            color=colors[path_name],
            linewidth=1.6,
            label=rf"$|H_{{{path_name}}}(f)|$",
        )
        peak_indices, _ = find_peaks(
            magnitude,
            prominence=max(float(np.max(magnitude)) * 0.03, np.finfo(float).eps),
        )
        ranked = sorted(peak_indices, key=lambda index: magnitude[index], reverse=True)[:5]
        for index in ranked:
            peaks_rows.append(
                {
                    "transfer_path": f"H_{path_name}",
                    "frequency_hz": float(band_frequency[index]),
                    "magnitude_v_per_n_assumed": float(magnitude[index]),
                    "phase_deg": float(np.angle(transfer[path_name][band][index], deg=True)),
                }
            )
        if ranked:
            strongest = ranked[0]
            magnitude_ax.scatter(
                [band_frequency[strongest]], [magnitude[strongest]],
                color=colors[path_name], s=38, zorder=3,
            )
            magnitude_ax.annotate(
                f"{band_frequency[strongest]:.1f} Hz",
                (band_frequency[strongest], magnitude[strongest]),
                xytext=(6, 7), textcoords="offset points",
            )

    magnitude_ax.set_title("Simplified transfer-function magnitude (unit-impulse assumption)")
    magnitude_ax.set_xlabel("Frequency [Hz]")
    magnitude_ax.set_ylabel("Magnitude [V/N, assumed]")
    magnitude_ax.set_xlim(FREQ_MIN_HZ, FREQ_MAX_HZ)
    magnitude_ax.set_ylim(bottom=0)
    magnitude_ax.grid(True, alpha=0.35)
    magnitude_ax.legend()
    magnitude_fig.tight_layout()
    magnitude_fig.savefig(OUTPUT_DIR / "simple_transfer_function_magnitude.png", dpi=200)
    plt.close(magnitude_fig)

    phase_fig, phase_ax = plt.subplots(figsize=(12, 6.5))
    for path_name in ["XZ", "YZ", "ZZ"]:
        phase_ax.plot(
            frequency[band],
            np.angle(transfer[path_name][band], deg=True),
            color=colors[path_name],
            linewidth=1.2,
            label=rf"$\angle H_{{{path_name}}}(f)$",
        )
    phase_ax.set_title("Simplified transfer-function phase (unit-impulse assumption)")
    phase_ax.set_xlabel("Frequency [Hz]")
    phase_ax.set_ylabel("Phase [deg]")
    phase_ax.set_xlim(FREQ_MIN_HZ, FREQ_MAX_HZ)
    phase_ax.set_ylim(-180, 180)
    phase_ax.grid(True, alpha=0.35)
    phase_ax.legend()
    phase_fig.tight_layout()
    phase_fig.savefig(OUTPUT_DIR / "simple_transfer_function_phase.png", dpi=200)
    plt.close(phase_fig)

    time_fig, time_ax = plt.subplots(figsize=(12, 5.5))
    for path_name, column in channel_map.items():
        baseline_mean = float(df.loc[baseline_mask, column].astype(float).mean())
        time_ax.plot(
            relative_time,
            response[column].to_numpy(dtype=float) - baseline_mean,
            color=colors[path_name], linewidth=1.0, label=f"{path_name[0]} output",
        )
    time_ax.set_title("Impulse responses used for the simplified transfer functions")
    time_ax.set_xlabel("Time from detected impact [s]")
    time_ax.set_ylabel("Load-cell output [V]")
    time_ax.grid(True, alpha=0.35)
    time_ax.legend()
    time_fig.tight_layout()
    time_fig.savefig(OUTPUT_DIR / "impulse_responses_xyz.png", dpi=200)
    plt.close(time_fig)

    peaks = pd.DataFrame(peaks_rows).sort_values(
        ["transfer_path", "magnitude_v_per_n_assumed"], ascending=[True, False]
    )
    peaks.to_csv(OUTPUT_DIR / "simple_transfer_function_peaks.csv", index=False, encoding="utf-8-sig")

    summary = (
        "Simplified transfer functions from 成功3回目.CSV\n"
        f"Assumed input: unit impulse in Z direction, F_Z(f)=1\n"
        f"Detected impact time: {impact_time:.7f} s\n"
        f"Analysis end time: {ANALYSIS_END_S:.7f} s\n"
        f"Samples: {len(response)}\n"
        f"Sampling frequency: {fs:.3f} Hz\n"
        f"Frequency-bin spacing: {fs / len(response):.3f} Hz\n\n"
        "Definitions:\n"
        "H_XZ(f) = FFT{v_X(t)}\n"
        "H_YZ(f) = FFT{v_Y(t)}\n"
        "H_ZZ(f) = FFT{v_Z(t)}\n\n"
        + peaks.to_string(index=False)
        + "\n"
    )
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
