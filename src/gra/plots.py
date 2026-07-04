import pylab as plt
import numpy as np

MAX_PLOT_POINTS = 8192


def _decimate_for_plot(x, y, max_points=MAX_PLOT_POINTS):
    n = len(x)
    if n <= max_points:
        return x, y
    step = max(1, n // max_points)
    return x[::step], y[::step]


def plot_strain(data):
    detectors = data.keys()
    fig, ax = plt.subplots(len(detectors), 1, figsize=(10, 2*len(detectors)), sharex=True)
    # Make sure ax is always a list, even if there's only one detector
    if len(detectors) == 1:
        ax = [ax]
    for i, det in enumerate(detectors):
        strain_data = data[det] # Time series object (gwpy)
        time, strain = _decimate_for_plot(strain_data.times.value, strain_data.value)
        ax[i].plot(time, strain, label=det)
        plt.xlim(time[0], time[-1])
        ax[i].set_ylabel('Strain')
        ax[i].legend()
    ax[-1].set_xlabel('Time (s)')
    return fig, ax
def plot_psd(psds, fig=None):
    detectors = sorted(psds.keys())
    if not detectors:
        raise ValueError("No PSD data to plot")
    if fig is None:
        fig, ax = plt.subplots(1, len(detectors), figsize=(10, 2 * len(detectors)), sharex=True)
        if len(detectors) == 1:
            ax = [ax]
    else:
        ax = list(fig.get_axes())
    for i, det in enumerate(detectors):
        f, psd = np.transpose(psds[det])
        ax[i].loglog(f, psd, label=det)
        fmin, fmax = 10, 2048
        ax[i].set_xlim(fmin, fmax)
        ax[i].set_ylim(1e-48, 1e-44)
        ax[i].set_ylabel('PSD')
        ax[i].legend()
    ax[-1].set_xlabel('Frequency (Hz)')
    return fig, ax


def save_figure(fig, path):
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

