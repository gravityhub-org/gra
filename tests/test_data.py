"""
Unit tests for gra.data (2MASS helpers).

File-system and network calls are mocked via tmp_path / monkeypatch so no
real downloads or data files are required.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


def test_public_api_on_submodules():
    from gra.data import get_2mass_data, get_lvk_strain, list_data_lvk, process_lvk_event
    from gra.data_lvk import h5_to_dict, remove_duplicates
    from gra.plots import plot_psd, plot_strain

    for fn in (
        get_2mass_data,
        get_lvk_strain,
        list_data_lvk,
        process_lvk_event,
        remove_duplicates,
        h5_to_dict,
        plot_strain,
        plot_psd,
    ):
        assert callable(fn)


def test_gra_package_import_stays_lightweight():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import gra\n"
        "heavy = any(m in sys.modules for m in ('gwpy', 'lalframe', 'h5py', 'bilby'))\n"
        "print('ok' if not heavy else 'heavy')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "ok"


def test_cli_import_stays_lightweight():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from gra.cli import app\n"
        "assert app is not None\n"
        "heavy = any(m in sys.modules for m in ('gwpy', 'lalframe', 'h5py', 'bilby'))\n"
        "print('ok' if not heavy else 'heavy')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# get_2mass_data routing
# ---------------------------------------------------------------------------

def test_get_2mass_data_routes_spectroscopic():
    from gra.data import get_2mass_data
    with patch("gra.data._get_2mass_spectroscopic", return_value=None) as mock:
        get_2mass_data("spectroscopic")
    mock.assert_called_once()


def test_get_2mass_data_individual_raises():
    from gra.data import get_2mass_data
    with pytest.raises(ValueError, match="Not implemented"):
        get_2mass_data("GW150914")


# ---------------------------------------------------------------------------
# _get_2mass_spectroscopic  (file already cached)
# ---------------------------------------------------------------------------

def test_get_2mass_spectroscopic_cached_no_return(tmp_path, monkeypatch):
    """When the CSV exists and return_data=False, return None without loading."""
    from gra import data as data_mod

    catalog_dir = tmp_path / "2mass"
    catalog_dir.mkdir()
    csv = catalog_dir / "2mass_galaxy_catalog_spec.csv"
    csv.write_text("col1,col2\n1,2\n")

    monkeypatch.chdir(tmp_path)

    result = data_mod._get_2mass_spectroscopic(return_data=False)
    assert result is None


def test_get_2mass_spectroscopic_cached_with_return(tmp_path, monkeypatch):
    """When the CSV exists and return_data=True, return the loaded table."""
    from astropy.table import Table
    from gra import data as data_mod

    catalog_dir = tmp_path / "2mass"
    catalog_dir.mkdir()
    csv = catalog_dir / "2mass_galaxy_catalog_spec.csv"
    # Write a minimal valid CSV that astropy can read
    csv.write_text("col1 col2\n1 2\n3 4\n")

    monkeypatch.chdir(tmp_path)

    mock_table = MagicMock(spec=Table)
    with patch("astropy.table.Table.read", return_value=mock_table):
        result = data_mod._get_2mass_spectroscopic(return_data=True)

    assert result is mock_table
