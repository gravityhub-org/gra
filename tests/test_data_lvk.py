"""
Unit tests for gra.data_lvk.

All external I/O (GWOSC, Zenodo, gwpy, frtools) is mocked so these tests
run fully offline without any data files.
"""

import io
import json
import os
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import typer


# ---------------------------------------------------------------------------
# remove_duplicates
# ---------------------------------------------------------------------------

def test_remove_duplicates_strips_version_suffix():
    from gra.data_lvk import remove_duplicates
    events = ["GW150914-v1", "GW150914-v2", "GW151226-v1"]
    assert remove_duplicates(events) == ["GW150914", "GW151226"]


def test_remove_duplicates_no_version():
    from gra.data_lvk import remove_duplicates
    events = ["GW150914", "GW151226"]
    assert remove_duplicates(events) == ["GW150914", "GW151226"]


def test_remove_duplicates_empty():
    from gra.data_lvk import remove_duplicates
    assert remove_duplicates([]) == []


# ---------------------------------------------------------------------------
# _pe_glob_pattern
# ---------------------------------------------------------------------------

def test_pe_glob_pattern_gwtc4():
    from gra.data_lvk import _pe_glob_pattern
    assert _pe_glob_pattern("GWTC-4.0", "GW200105") == "*GW200105*"


def test_pe_glob_pattern_gwtc5():
    from gra.data_lvk import _pe_glob_pattern
    assert _pe_glob_pattern("GWTC-5.0", "GW240406_062847") == "*GW240406_062847*"


def test_pe_zenodo_record_ids_gwtc5():
    from gra.data_lvk import _pe_zenodo_record_ids
    assert _pe_zenodo_record_ids("GWTC-5.0") == ["20348005", "20348006"]


def test_pe_zenodo_record_ids_single():
    from gra.data_lvk import _pe_zenodo_record_ids
    assert _pe_zenodo_record_ids("GWTC-4.0") == ["17014085"]
def test_pe_glob_pattern_older_catalog():
    from gra.data_lvk import _pe_glob_pattern
    assert _pe_glob_pattern("GWTC-3-confident", "GW190814") == "*GW190814*nocosmo*.h5"


# ---------------------------------------------------------------------------
# _ensure_dir
# ---------------------------------------------------------------------------

def test_ensure_dir_creates(tmp_path):
    from gra.data_lvk import _ensure_dir
    target = tmp_path / "new_dir"
    assert not target.exists()
    _ensure_dir(str(target))
    assert target.is_dir()


def test_ensure_dir_idempotent(tmp_path):
    from gra.data_lvk import _ensure_dir
    _ensure_dir(str(tmp_path))   # already exists — must not raise
    assert tmp_path.is_dir()


# ---------------------------------------------------------------------------
# h5_to_dict
# ---------------------------------------------------------------------------

def test_h5_to_dict_datasets(tmp_path):
    from gra.data_lvk import h5_to_dict
    fpath = tmp_path / "test.hdf5"
    with h5py.File(fpath, "w") as f:
        f.create_dataset("mass_1", data=np.array([30.0, 31.0]))
        f.create_dataset("mass_2", data=np.array([25.0, 26.0]))

    with h5py.File(fpath, "r") as f:
        result = h5_to_dict(f)

    np.testing.assert_array_equal(result["mass_1"], [30.0, 31.0])
    np.testing.assert_array_equal(result["mass_2"], [25.0, 26.0])


def test_h5_to_dict_nested_groups(tmp_path):
    from gra.data_lvk import h5_to_dict
    fpath = tmp_path / "nested.hdf5"
    with h5py.File(fpath, "w") as f:
        g = f.create_group("posterior_samples")
        g.create_dataset("chi_eff", data=np.array([0.1]))

    with h5py.File(fpath, "r") as f:
        result = h5_to_dict(f)

    assert "posterior_samples" in result
    np.testing.assert_array_equal(result["posterior_samples"]["chi_eff"], [0.1])


# ---------------------------------------------------------------------------
# check_event_name
# ---------------------------------------------------------------------------

def test_check_event_name_valid():
    from gra.data_lvk import check_event_name
    with patch("gra.data_lvk._list_lvk_data", return_value=["GW150914", "GW151226"]):
        events = check_event_name("GW150914")
    assert "GW150914" in events


def test_check_event_name_invalid_raises():
    from gra.data_lvk import check_event_name
    import click
    with patch("gra.data_lvk._list_lvk_data", return_value=["GW150914"]):
        with pytest.raises((SystemExit, click.exceptions.Exit, typer.Exit)):
            check_event_name("GW999999")


# ---------------------------------------------------------------------------
# _find_event_catalog
# ---------------------------------------------------------------------------

def test_find_event_catalog_found():
    from gra.data_lvk import _find_event_catalog
    def fake_find_datasets(type, catalog):
        if catalog == "GWTC-3-confident":
            return ["GW190814-v1"]
        return []
    with patch("gra.data_lvk.find_datasets", side_effect=fake_find_datasets):
        result = _find_event_catalog("GW190814")
    assert result == "GWTC-3-confident"


def test_find_event_catalog_not_found():
    from gra.data_lvk import _find_event_catalog
    with patch("gra.data_lvk.find_datasets", return_value=[]):
        result = _find_event_catalog("GW999999")
    assert result is None


def test_find_event_catalog_gwtc5():
    from gra.data_lvk import _find_event_catalog

    def fake_find_datasets(type, catalog):
        if catalog == "GWTC-5.0":
            return ["GW240406_062847-v1"]
        return []

    with patch("gra.data_lvk.find_datasets", side_effect=fake_find_datasets):
        result = _find_event_catalog("GW240406_062847")
    assert result == "GWTC-5.0"


# ---------------------------------------------------------------------------
# _get_lvk_info_individual  (cache hit path)
# ---------------------------------------------------------------------------

def test_get_lvk_info_individual_loads_cache(tmp_path, monkeypatch):
    from gra.data_lvk import _get_lvk_info_individual

    event = "GW150914"
    event_dir = tmp_path / event
    event_dir.mkdir()
    cache = {"event_name": event, "gps": 1126259462.4, "detectors": ["H1", "L1"]}
    (event_dir / f"{event}_info.json").write_text(json.dumps(cache))

    monkeypatch.chdir(tmp_path)

    info = _get_lvk_info_individual(event)

    assert info["gps"] == 1126259462.4
    assert info["detectors"] == {"H1", "L1"}
