"""
GL980 hammering-test analysis
- Reads rows 21–1020 of a GL980 CSV
- Draws three time histories
- Draws full-record FFTs
- Draws FFTs over an arbitrary time interval

Required packages:
    pip install numpy pandas matplotlib scipy
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks, windows


# =========================
# User settings
# =========================
CSV_PATH = Path("成功4回目.CSV")

# Convert voltage to force only when:
# CAL = 1000 microstrain and VOLTAGE OUT = 5.00 V
# For LSM-B-50NSA1: 50 N <-> 5 V, therefore 10 N/V.
CONVERT_TO_FORCE = False
FORCE_PER_VOLT = 10.0  # N/V

# Arbitrary interval used for section FFT [s].
# Change these values after checking the time-history plots.
FFT_START_S = 0.0975
FFT_END_S = 0.1422

# Frequency range shown in FFT plots.
# For the current 1000 Hz sampling data, this cannot exceed 500 Hz.
FFT_DISPLAY_MAX_HZ = 1500.0

# Number of peaks printed for each axis.
NUMBER_OF_PEAKS = 5

OUTPUT_DIR = Path("hammering_results")


def read_gl980_csv(csv_path: Path) -> pd.DataFrame:
    """
    Reads the GL980 CSV.
    """

    df = pd.read_csv(
        csv_path,
        skiprows=20,
        header=None,
        encoding="utf-8-sig",
    )

    # 最初の9列だけ使用
    df = df.iloc[:, :9]

    df.columns = [
        "number",
        "date",
        "clock_time",
        "time_us",
        "x_v",
        "y_v",
        "z_v",
        "alarm",
        "alarm_out",
    ]

    numeric_columns = [
        "time_us",
        "x_v",
        "y_v",
        "z_v",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=numeric_columns
    ).reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(
            "有効な測定データが2点未満です。CSVの行構成を確認してください。"
        )

    df["time_s"] = df["time_us"] * 1.0e-6

    return df


def sampling_information(time_s: np.ndarray) -> tuple[float, float, float]:
    """Returns sampling interval, sampling frequency, and Nyquist frequency."""
    differences = np.diff(time_s)

    if np.any(differences <= 0):
        raise ValueError("時刻データが単調増加していません。")

    dt = float(np.median(differences))
    fs = 1.0 / dt
    nyquist = fs / 2.0

    # Warn when time steps are not sufficiently uniform.
    relative_variation = np.max(np.abs(differences - dt)) / dt
    if relative_variation > 1.0e-3:
        print("警告：サンプリング間隔が一定ではありません。FFT結果を慎重に確認してください。")

    return dt, fs, nyquist


def one_sided_fft(
    signal: np.ndarray,
    fs: float,
    use_hann_window: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates a one-sided amplitude spectrum.

    Processing:
    1. Remove the mean.
    2. Optionally apply a Hann window.
    3. Correct amplitude for window coherent gain.
    """
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 4:
        raise ValueError("FFT区間のデータ数が少なすぎます。")

    signal = detrend(signal, type="constant")

    if use_hann_window:
        window = windows.hann(len(signal), sym=False)
    else:
        window = np.ones(len(signal))

    transformed = np.fft.rfft(signal * window)
    frequency = np.fft.rfftfreq(len(signal), d=1.0 / fs)

    # One-sided amplitude spectrum, with window-gain correction.
    amplitude = 2.0 * np.abs(transformed) / np.sum(window)

    # DC and Nyquist bins should not be doubled.
    amplitude[0] *= 0.5
    if len(signal) % 2 == 0:
        amplitude[-1] *= 0.5

    return frequency, amplitude


def select_interval(
    df: pd.DataFrame,
    start_s: float,
    end_s: float,
) -> pd.DataFrame:
    """Extracts an arbitrary interval and resets its time origin to zero."""
    if end_s <= start_s:
        raise ValueError("FFT_END_SはFFT_START_Sより大きくしてください。")

    section = df[(df["time_s"] >= start_s) & (df["time_s"] <= end_s)].copy()

    if len(section) < 4:
        raise ValueError("指定したFFT区間に十分なデータがありません。")

    section["time_s_relative"] = section["time_s"] - section["time_s"].iloc[0]
    return section


def plot_time_histories(
    df: pd.DataFrame,
    signal_columns: dict[str, str],
    unit: str,
    output_dir: Path,
) -> None:
    """Creates one time-history figure for each axis."""

    # 指定時刻に最も近いデータ点を取得
    start_index = (df["time_s"] - FFT_START_S).abs().idxmin()
    end_index = (df["time_s"] - FFT_END_S).abs().idxmin()

    actual_start_time = df.loc[start_index, "time_s"]
    actual_end_time = df.loc[end_index, "time_s"]

    for column, axis_name in signal_columns.items():
        start_value = df.loc[start_index, column]
        end_value = df.loc[end_index, column]

        fig, ax = plt.subplots(figsize=(10, 5))

        # 時刻歴
        ax.plot(
            df["time_s"],
            df[column],
            linewidth=1.0,
            label=f"{axis_name}-axis output",
        )

        # FFT対象区間を薄く表示
        ax.axvspan(
            actual_start_time,
            actual_end_time,
            alpha=0.2,
            label="Selected FFT interval",
        )

        # FFT開始点
        ax.plot(
            actual_start_time,
            start_value,
            marker="o",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=2,
            linestyle="none",
            label="FFT start",
        )

        # FFT終了点
        ax.plot(
            actual_end_time,
            end_value,
            marker="o",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=2,
            linestyle="none",
            label="FFT end",
        )

        # 開始・終了時刻の注記
        ax.annotate(
            f"Start: {actual_start_time:.4f} s",
            xy=(actual_start_time, start_value),
            xytext=(8, 12),
            textcoords="offset points",
        )

        ax.annotate(
            f"End: {actual_end_time:.4f} s",
            xy=(actual_end_time, end_value),
            xytext=(8, -18),
            textcoords="offset points",
        )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{axis_name}-axis output [{unit}]")
        ax.set_title(f"{axis_name}-axis time history")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            output_dir / f"time_history_{axis_name.lower()}.png",
            dpi=200,
        )
        plt.close(fig)


def plot_fft(
    df: pd.DataFrame,
    signal_columns: dict[str, str],
    fs: float,
    unit: str,
    output_dir: Path,
    filename_suffix: str,
    title_suffix: str,
) -> None:
    """Creates one FFT figure for each axis and prints major spectral peaks."""
    for column, axis_name in signal_columns.items():
        frequency, amplitude = one_sided_fft(df[column].to_numpy(), fs)

        mask = (frequency > 0.0) & (frequency <= FFT_DISPLAY_MAX_HZ)
        displayed_frequency = frequency[mask]
        displayed_amplitude = amplitude[mask]

        if len(displayed_frequency) == 0:
            raise ValueError("FFT_DISPLAY_MAX_HZの設定を確認してください。")

        # Peak detection. Prominence threshold suppresses very small noise peaks.
        prominence = max(np.max(displayed_amplitude) * 0.03, np.finfo(float).eps)
        peak_indices, properties = find_peaks(
            displayed_amplitude,
            prominence=prominence,
        )

        ranked_peaks = sorted(
            peak_indices,
            key=lambda index: displayed_amplitude[index],
            reverse=True,
        )[:NUMBER_OF_PEAKS]

        print(f"\n{axis_name}-axis {title_suffix}:")
        if ranked_peaks:
            for index in ranked_peaks:
                print(
                    f"  {displayed_frequency[index]:9.3f} Hz, "
                    f"amplitude = {displayed_amplitude[index]:.6g} {unit}"
                )
        else:
            print("  明確なピークを検出できませんでした。")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(displayed_frequency, displayed_amplitude, linewidth=1.0)

        for index in ranked_peaks:
            ax.plot(
                displayed_frequency[index],
                displayed_amplitude[index],
                marker="o",
            )
            ax.annotate(
                f"{displayed_frequency[index]:.1f} Hz",
                (
                    displayed_frequency[index],
                    displayed_amplitude[index],
                ),
                xytext=(5, 5),
                textcoords="offset points",
            )

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(f"Single-sided amplitude [{unit}]")
        ax.set_title(f"{axis_name}-axis FFT {title_suffix}")
        ax.set_xlim(0.0, FFT_DISPLAY_MAX_HZ)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"fft_{axis_name.lower()}_{filename_suffix}.png",
            dpi=200,
        )
        plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = read_gl980_csv(CSV_PATH)
    dt, fs, nyquist = sampling_information(df["time_s"].to_numpy())

    print(f"Data points           : {len(df)}")
    print(f"Measurement duration  : {df['time_s'].iloc[-1] - df['time_s'].iloc[0]:.6f} s")
    print(f"Sampling interval     : {dt:.9f} s")
    print(f"Sampling frequency    : {fs:.3f} Hz")
    print(f"Nyquist frequency     : {nyquist:.3f} Hz")
    print(f"Full-record resolution: {fs / len(df):.3f} Hz")

    # Allow a very small floating-point tolerance at the Nyquist limit.
    if FFT_DISPLAY_MAX_HZ > nyquist + 1.0e-9:
        raise ValueError(
            f"FFT_DISPLAY_MAX_HZ={FFT_DISPLAY_MAX_HZ} Hz exceeds "
            f"the Nyquist frequency {nyquist:.3f} Hz."
        )

    if CONVERT_TO_FORCE:
        df["x"] = df["x_v"] * FORCE_PER_VOLT
        df["y"] = df["y_v"] * FORCE_PER_VOLT
        df["z"] = df["z_v"] * FORCE_PER_VOLT
        unit = "N"
        signal_columns = {"x": "X", "y": "Y", "z": "Z"}
    else:
        unit = "V"
        signal_columns = {"x_v": "X", "y_v": "Y", "z_v": "Z"}

    # 1. Three time-history plots.
    plot_time_histories(df, signal_columns, unit, OUTPUT_DIR)

    # 2. FFT over the entire record.
    plot_fft(
        df=df,
        signal_columns=signal_columns,
        fs=fs,
        unit=unit,
        output_dir=OUTPUT_DIR,
        filename_suffix="full",
        title_suffix="(full record)",
    )

    # 3. FFT over an arbitrary interval.
    section = select_interval(df, FFT_START_S, FFT_END_S)
    section_duration = (
        section["time_s"].iloc[-1] - section["time_s"].iloc[0]
    )
    print(
        f"\nSelected FFT interval : {FFT_START_S:.6f}–{FFT_END_S:.6f} s"
    )
    print(f"Selected data points  : {len(section)}")
    print(f"Section duration      : {section_duration:.6f} s")
    print(
        f"Section resolution    : "
        f"{fs / len(section):.3f} Hz"
    )

    plot_fft(
        df=section,
        signal_columns=signal_columns,
        fs=fs,
        unit=unit,
        output_dir=OUTPUT_DIR,
        filename_suffix="selected",
        title_suffix=(
            f"({FFT_START_S:.4f}–{FFT_END_S:.4f} s)"
        ),
    )

    print(f"\nFigures were saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
