"""PE / likelihood helpers: event_name paths here; core logic in ``likelihoodloader``."""


from likelihoodloader import (
    build_interferometers as _build_interferometers_from_files,
    build_likelihood as _build_likelihood_from_files,
    build_waveform_generator as _build_waveform_generator_from_posterior,
    load_and_crop_strain as _load_and_crop_strain,
    read_calibration_envelope_from_posterior as _read_calib,
    read_config_from_posterior as _read_config,
    read_frequency_cuts_from_posterior as _read_freq_cuts,
    read_meta_data_from_posterior as _read_meta,
    read_prior_from_posterior as _read_prior,
    load_posterior_dict,
    load_psds_from_posterior,
)

from . import data_lvk


def _posterior_path(event_name: str) -> str:
    return data_lvk._get_lvk_pe_data_filename(event_name)


def _strain_gwf_map(event_name: str) -> dict[str, str]:
    info = data_lvk._get_lvk_info_individual(event_name)
    dets = list(info["detectors"])
    return {
        d: f"{event_name}/{event_name}_{d}_strain.gwf" for d in dets
    }


def read_calibration_envelope_from_posterior(event_name: str):
    return _read_calib(_posterior_path(event_name))


def read_prior_from_posterior(event_name: str):
    return _read_prior(_posterior_path(event_name))


def read_config_from_posterior(event_name: str):
    return _read_config(_posterior_path(event_name))


def read_meta_data_from_posterior(event_name: str):
    return _read_meta(_posterior_path(event_name))


def read_frequency_cuts_from_posterior(event_name: str):
    return _read_freq_cuts(_posterior_path(event_name))


def load_and_crop_strain(event_name: str):
    return _load_and_crop_strain(
        _posterior_path(event_name), _strain_gwf_map(event_name)
    )


def build_interferometers(event_name: str):
    path = _posterior_path(event_name)
    pdict = load_posterior_dict(path)
    data = _load_and_crop_strain(
        path, _strain_gwf_map(event_name), posterior_dict=pdict
    )
    return _build_interferometers_from_files(
        path, data, psds=load_psds_from_posterior(path, pdict), posterior_dict=pdict
    )


def build_waveform_generator(event_name: str):
    return _build_waveform_generator_from_posterior(_posterior_path(event_name))


def build_likelihood(event_name: str):
    return _build_likelihood_from_files(
        _posterior_path(event_name), _strain_gwf_map(event_name)
    )

if __name__ == "__main__":
    event_name = "GW231123_135430"
    _ = data_lvk._get_lvk_info_individual(event_name)
    ppath = _posterior_path(event_name)
    _ = load_posterior_dict(ppath)
    _ = _read_config(ppath)
    _ = load_psds_from_posterior(ppath)
    _ = _read_calib(ppath)
    _ = _read_prior(ppath)
    _ = data_lvk.get_lvk_strain_individual_sync(event_name)
    _ = _load_and_crop_strain(ppath, _strain_gwf_map(event_name))
    _ = build_interferometers(event_name)
    _ = _build_waveform_generator_from_posterior(ppath)
    _ = build_likelihood(event_name)

