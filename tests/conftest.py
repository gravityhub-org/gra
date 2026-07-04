"""Shared fixtures for gra tests (offline, cwd-relative event layout)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

matplotlib.use("Agg")

EVENT_NAME = "GW150914"
EVENT_GPS = 1126259462.4
DETECTORS = ("H1", "L1")
APPROXIMANT = "C00:IMRPhenomXPHM"
STRAIN_DURATION_S = 128
SAMPLE_RATE = 4096


def _make_strain_timeseries(det: str, gps: float = EVENT_GPS) -> TimeSeries:
    n = int(STRAIN_DURATION_S * SAMPLE_RATE)
    times = np.arange(n) / SAMPLE_RATE + (gps - STRAIN_DURATION_S / 2)
    data = np.random.randn(n) * 1e-23
    ts = TimeSeries(data, times=times, sample_rate=SAMPLE_RATE)
    ts.channel = f"{det}:GWOSC-STRAIN"
    return ts


def _write_strain_gwfs(event_dir: Path, event_name: str, detectors=DETECTORS) -> None:
    for det in detectors:
        ts = _make_strain_timeseries(det)
        path = event_dir / f"{event_name}_{det}_strain.gwf"
        ts.write(str(path), format="gwf")


def _write_event_info(event_dir: Path, event_name: str, detectors=DETECTORS) -> dict:
    info = {
        "event_name": event_name,
        "gps": EVENT_GPS,
        "detectors": list(detectors),
    }
    (event_dir / f"{event_name}_info.json").write_text(json.dumps(info, indent=4))
    return info


def _write_pe_file(event_dir: Path, detectors=DETECTORS, approximant: str = APPROXIMANT) -> Path:
    pe_dir = event_dir / "official_pe"
    pe_dir.mkdir(parents=True, exist_ok=True)
    pe_path = pe_dir / f"{EVENT_NAME}_pe.hdf5"
    freqs = np.logspace(1, 2.5, 64)
    with h5py.File(pe_path, "w") as f:
        group = f.create_group(approximant)
        psds = group.create_group("psds")
        for det in detectors:
            psd_vals = np.full_like(freqs, 1e-46)
            psds.create_dataset(det, data=np.column_stack([freqs, psd_vals]))
    return pe_path


@pytest.fixture
def event_workspace(tmp_path) -> Path:
    """Minimal on-disk event tree: info.json + strain GWF per detector."""
    event_dir = tmp_path / EVENT_NAME
    event_dir.mkdir()
    _write_event_info(event_dir, EVENT_NAME)
    _write_strain_gwfs(event_dir, EVENT_NAME)
    return tmp_path


@pytest.fixture
def event_workspace_with_pe(event_workspace) -> Path:
    """Event tree with official PE HDF5 (PSDs for Welch/official plots)."""
    event_dir = event_workspace / EVENT_NAME
    _write_pe_file(event_dir)
    return event_workspace
