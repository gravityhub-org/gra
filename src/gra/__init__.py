"""Gravitational-wave research assistant."""

from . import data
from . import data_lvk
from . import plots

from .data import get_2mass_data, get_lvk_strain, list_data_lvk, process_lvk_event
from .data_lvk import h5_to_dict, remove_duplicates
from .plots import plot_psd, plot_strain

__all__ = [
    "data",
    "data_lvk",
    "plots",
    "get_2mass_data",
    "get_lvk_strain",
    "list_data_lvk",
    "process_lvk_event",
    "remove_duplicates",
    "h5_to_dict",
    "plot_strain",
    "plot_psd",
]
