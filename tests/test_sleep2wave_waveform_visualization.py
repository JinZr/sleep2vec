from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sleep2wave.visualization.downstream_eval_plots import render_waveform_example_plot


def test_waveform_example_plot_expands_high_rate_samples():
    duration_sec = 900
    low_rate = 4
    high_rate = 128
    low_waveform = np.linspace(0.0, 1.0, duration_sec * low_rate, dtype=np.float32)
    high_waveform = np.linspace(0.0, 1.0, duration_sec * high_rate, dtype=np.float32)

    low_fig = render_waveform_example_plot(
        low_waveform,
        low_waveform,
        sample_rate_hz=low_rate,
        title="low rate",
        generated_label="Reconstruction",
    )
    high_fig = render_waveform_example_plot(
        high_waveform,
        high_waveform,
        sample_rate_hz=high_rate,
        title="high rate",
        generated_label="Reconstruction",
    )

    try:
        assert low_fig.get_size_inches()[0] == 10.5
        assert low_fig.axes[0].lines[0].get_xdata().size <= 2000
        assert high_fig.get_size_inches()[0] > low_fig.get_size_inches()[0]
        assert high_fig.axes[0].lines[0].get_xdata().size > low_fig.axes[0].lines[0].get_xdata().size * 2
    finally:
        plt.close(low_fig)
        plt.close(high_fig)
