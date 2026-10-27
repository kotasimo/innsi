"""Check library interoperability using synthetic data, not research analysis."""
import io
import os
from pathlib import Path
import tempfile

cache = Path(tempfile.gettempdir()) / f"loadcell-cache-{os.getuid()}"
os.environ.setdefault("XDG_CACHE_HOME", str(cache))
os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


def main():
    frame = pd.read_csv(io.StringIO("time,value\n0,1\n1,0\n2,-1\n3,0\n"))
    values = frame["value"].to_numpy(dtype=float)
    spectrum = np.fft.rfft(values)
    np.testing.assert_allclose(np.abs(spectrum), [0, 2, 0])
    np.testing.assert_allclose(signal.detrend(values, type="constant"), values)
    with tempfile.TemporaryDirectory(prefix="loadcell-check-") as directory:
        path = Path(directory) / "check.png"
        fig, ax = plt.subplots()
        ax.plot(frame["time"], values)
        fig.savefig(path)
        plt.close(fig)
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    print("OK: NumPy FFT, SciPy signal processing, Pandas CSV, Matplotlib PNG")


if __name__ == "__main__":
    main()
