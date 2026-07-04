"""
End-to-end tests for gra.

Exercises full call chains (CLI → data → data_lvk → plots, pe_lvk → likelihoodloader)
using on-disk fixtures. Only external services (GWOSC catalog listing) are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from typer.testing import CliRunner

from gra.cli import app

EVENT_NAME = "GW150914"
EVENT_GPS = 1126259462.4
DETECTORS = ("H1", "L1")
APPROXIMANT = "C00:IMRPhenomXPHM"

runner = CliRunner()


# ---------------------------------------------------------------------------
# Package → data → data_lvk
# ---------------------------------------------------------------------------

def test_get_lvk_strain_e2e_cached(event_workspace, monkeypatch):
    monkeypatch.chdir(event_workspace)

    from gra.data import get_lvk_strain

    name, strain = get_lvk_strain(EVENT_NAME, download_pe=False)

    assert name == EVENT_NAME
    assert set(strain) == set(DETECTORS)
    for det in DETECTORS:
        assert len(strain[det]) > 0


def test_process_lvk_event_e2e_writes_artifacts(event_workspace_with_pe, monkeypatch):
    monkeypatch.chdir(event_workspace_with_pe)
    event_dir = event_workspace_with_pe / EVENT_NAME

    from gra.data import process_lvk_event

    process_lvk_event(EVENT_NAME)

    assert (event_dir / f"{EVENT_NAME}_strain.pdf").is_file()
    assert (event_dir / f"{EVENT_NAME}_psd.pdf").is_file()
    assert (event_dir / f"{EVENT_NAME}_psd_welch.pdf").is_file()
    for det in DETECTORS:
        assert (event_dir / f"{EVENT_NAME}_{det}_psd.npy").is_file()
        assert (event_dir / f"{EVENT_NAME}_{det}_before_psd_welch_seglen_4s.npy").is_file()
        assert (event_dir / f"{EVENT_NAME}_{det}_after_psd_welch_seglen_4s.npy").is_file()
        assert (event_dir / f"{EVENT_NAME}_{det}_noise_before.gwf").is_file()
        assert (event_dir / f"{EVENT_NAME}_{det}_noise_after.gwf").is_file()


# ---------------------------------------------------------------------------
# CLI → data (no gra-layer mocks)
# ---------------------------------------------------------------------------

def test_cli_get_lvk_e2e_cached(event_workspace, monkeypatch):
    monkeypatch.chdir(event_workspace)

    result = runner.invoke(app, ["data", "get", "lvk", EVENT_NAME, "--no-pe"])

    assert result.exit_code == 0


def test_cli_process_lvk_e2e(event_workspace_with_pe, monkeypatch):
    monkeypatch.chdir(event_workspace_with_pe)
    event_dir = event_workspace_with_pe / EVENT_NAME

    result = runner.invoke(app, ["data", "process", "lvk", EVENT_NAME])

    assert result.exit_code == 0
    assert (event_dir / f"{EVENT_NAME}_strain.pdf").exists()


def test_cli_ls_lvk_e2e(monkeypatch):
    def fake_find_datasets(type, catalog=None):
        if type == "events" and catalog == "GWTC-1-confident":
            return [f"{EVENT_NAME}-v1"]
        return []

    with patch("gra.data_lvk.find_datasets", side_effect=fake_find_datasets):
        result = runner.invoke(app, ["data", "ls", "lvk"])

    assert result.exit_code == 0
    assert EVENT_NAME in result.stdout


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def test_plots_strain_and_psd_e2e(tmp_path, event_workspace):
    from gra.plots import plot_psd, plot_strain, save_figure
    from gra.data_lvk import _read_gwf

    event_dir = event_workspace / EVENT_NAME
    strain = {
        det: _read_gwf(str(event_dir / f"{EVENT_NAME}_{det}_strain.gwf"))
        for det in DETECTORS
    }

    fig, _ = plot_strain(strain)
    strain_pdf = tmp_path / "strain.pdf"
    save_figure(fig, strain_pdf)
    assert strain_pdf.stat().st_size > 0

    freqs = np.logspace(1, 2.5, 64)
    psds = {
        det: np.column_stack([freqs, np.full_like(freqs, 1e-46)])
        for det in DETECTORS
    }
    fig, _ = plot_psd(psds)
    psd_pdf = tmp_path / "psd.pdf"
    save_figure(fig, psd_pdf)
    assert psd_pdf.stat().st_size > 0


# ---------------------------------------------------------------------------
# pe_lvk → likelihoodloader path wiring
# ---------------------------------------------------------------------------

def test_pe_lvk_posterior_and_strain_paths(event_workspace_with_pe, monkeypatch):
    monkeypatch.chdir(event_workspace_with_pe)

    from gra.pe_lvk import _posterior_path, _strain_gwf_map

    pe_path = _posterior_path(EVENT_NAME)
    strain_map = _strain_gwf_map(EVENT_NAME)

    assert pe_path.endswith(".hdf5")
    assert Path(pe_path).is_file()
    assert set(strain_map) == set(DETECTORS)
    for det in DETECTORS:
        assert strain_map[det] == f"{EVENT_NAME}/{EVENT_NAME}_{det}_strain.gwf"
        assert Path(strain_map[det]).is_file()


def test_pe_lvk_build_likelihood_e2e(event_workspace_with_pe, monkeypatch):
    monkeypatch.chdir(event_workspace_with_pe)

    captured = {}

    def fake_build(posterior_path, strain_map):
        captured["posterior_path"] = posterior_path
        captured["strain_map"] = strain_map
        return "mock-likelihood"

    with patch("gra.pe_lvk._build_likelihood_from_files", side_effect=fake_build):
        from gra.pe_lvk import build_likelihood

        result = build_likelihood(EVENT_NAME)

    assert result == "mock-likelihood"
    assert captured["posterior_path"].endswith(".hdf5")
    assert set(captured["strain_map"]) == set(DETECTORS)


def test_pe_lvk_read_config_e2e(event_workspace_with_pe, monkeypatch):
    monkeypatch.chdir(event_workspace_with_pe)

    sentinel = {"approximant": APPROXIMANT}

    with patch("gra.pe_lvk._read_config", return_value=sentinel) as mock_read:
        from gra.pe_lvk import read_config_from_posterior

        result = read_config_from_posterior(EVENT_NAME)

    assert result == sentinel
    mock_read.assert_called_once()
    assert mock_read.call_args[0][0].endswith(".hdf5")


def test_pe_lvk_build_interferometers_e2e(event_workspace_with_pe, monkeypatch):
    monkeypatch.chdir(event_workspace_with_pe)

    with (
        patch("gra.pe_lvk.load_posterior_dict", return_value={"meta": True}),
        patch("gra.pe_lvk._load_and_crop_strain", return_value={"H1": "data"}),
        patch("gra.pe_lvk.load_psds_from_posterior", return_value={"H1": []}),
        patch(
            "gra.pe_lvk._build_interferometers_from_files",
            return_value="mock-ifos",
        ) as mock_build,
    ):
        from gra.pe_lvk import build_interferometers

        result = build_interferometers(EVENT_NAME)

    assert result == "mock-ifos"
    posterior_path, data, kwargs = mock_build.call_args[0][0], mock_build.call_args[0][1], mock_build.call_args[1]
    assert posterior_path.endswith(".hdf5")
    assert data == {"H1": "data"}
    assert kwargs["psds"] == {"H1": []}
    assert kwargs["posterior_dict"] == {"meta": True}


# ---------------------------------------------------------------------------
# data.py facade
# ---------------------------------------------------------------------------

def test_data_facade_matches_data_lvk_e2e(event_workspace, monkeypatch):
    monkeypatch.chdir(event_workspace)

    from gra import data
    from gra import data_lvk

    name, strain = data.get_lvk_strain(EVENT_NAME, download_pe=False)
    info = data_lvk._get_lvk_info_individual(EVENT_NAME)

    assert name == EVENT_NAME
    assert info["gps"] == EVENT_GPS
    assert set(strain) == set(info["detectors"])
