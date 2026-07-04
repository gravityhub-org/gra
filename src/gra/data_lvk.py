# Get rid of annoying warning about swiglal redir stdio:
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import json
import numpy as np
import h5py
import gwpy
from gwosc.datasets import find_datasets, event_gps, event_detectors
from zenodo_get import download as zenodo_download
import asyncio
import concurrent.futures
from functools import partial
import gwpy.timeseries
from lalframe.utils import frtools
import os

import typer
from rich.console import Console

console = Console()

pe_zenodo_releases = {}
pe_zenodo_releases['GWTC-2.1-confident'] = 'https://zenodo.org/records/6513631'
pe_zenodo_releases['GWTC-3-confident'] = 'https://zenodo.org/records/5546663'
pe_zenodo_releases['GWTC-4.0'] = 'https://zenodo.org/records/17014085'
pe_zenodo_releases['GWTC-5.0'] = [
    'https://zenodo.org/records/20348005',
    'https://zenodo.org/records/20348006',
]

_LVK_CATALOGS = [
    'GWTC-1-confident',
    'GWTC-2.1-confident',
    'GWTC-3-confident',
    'GWTC-4.0',
    'GWTC-5.0',
    'O4_Discovery_Papers',
]
_lvk_events_cache = None
_event_catalog_cache = {}


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _strain_channel(filename):
    """Pick the GWOSC strain channel, not DQ/mask channels listed first in O4 files."""
    for channel in frtools.get_channels(filename):
        if 'GWOSC-STRAIN' in channel or channel.endswith(':STRAIN'):
            return channel
    return frtools.get_channels(filename)[0]


def _read_gwf(filename):
    """Read a GWF file, inferring the channel name automatically."""
    return gwpy.timeseries.TimeSeries.read(
        filename, format='gwf', channel=_strain_channel(filename),
    )


def _read_gwfs(paths_by_key):
    """Read multiple GWF files in parallel using a process per file."""
    if len(paths_by_key) <= 1:
        return {key: _read_gwf(path) for key, path in paths_by_key.items()}
    items = list(paths_by_key.items())
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(items)) as executor:
        results = list(executor.map(_read_gwf, (path for _, path in items)))
    return {key: result for (key, _), result in zip(items, results)}


def _process_gwf_paths(event_name, info):
    """Return all on-disk GWF paths needed for processing one event."""
    paths = {}
    for det in info['detectors']:
        paths[(det, 'strain')] = _strain_gwf_path(event_name, det)
        before = f"{event_name}/{event_name}_{det}_noise_before.gwf"
        after = f"{event_name}/{event_name}_{det}_noise_after.gwf"
        if os.path.exists(before) and os.path.exists(after):
            paths[(det, 'before')] = before
            paths[(det, 'after')] = after
    return paths


def _split_process_data(loaded, detectors):
    strain_data = {}
    noise = {}
    for det in detectors:
        if (det, 'strain') in loaded:
            strain_data[det] = loaded[(det, 'strain')]
        before_key = (det, 'before')
        after_key = (det, 'after')
        if before_key in loaded and after_key in loaded:
            noise[det] = {'before': loaded[before_key], 'after': loaded[after_key]}
    return strain_data, noise


def _ensure_lvk_catalog_cache():
    global _lvk_events_cache
    if _lvk_events_cache is not None:
        return
    events_all = []
    for catalog in _LVK_CATALOGS:
        events = remove_duplicates(find_datasets(type='events', catalog=catalog))
        events_all.extend(events)
    _lvk_events_cache = events_all


def _find_event_catalog(event_name):
    """Return the catalog name that contains *event_name*, or None."""
    if event_name in _event_catalog_cache:
        return _event_catalog_cache[event_name]
    for cat in list(pe_zenodo_releases.keys()) + ['O4_Discovery_Papers']:
        if any(event_name in e for e in find_datasets(type='event', catalog=cat)):
            _event_catalog_cache[event_name] = cat
            return cat
    return None


def _pe_zenodo_record_ids(catalog):
    entry = pe_zenodo_releases[catalog]
    if isinstance(entry, str):
        entry = [entry]
    return [url.rstrip('/').split('/')[-1] for url in entry]


def _pe_glob_pattern(catalog, event_name):
    """Return the Zenodo file-glob pattern for a given catalog and event."""
    if catalog in ("GWTC-4.0", "GWTC-5.0"):
        return f"*{event_name}*"
    return f"*{event_name}*nocosmo*.h5"


async def _get_lvk_pe_data_async(event_name):
    output_dir = f"{os.getcwd()}/{event_name}"
    if any(fname.endswith('.hdf5') for fname in os.listdir(output_dir)):
        typer.echo(f"PE data for '{event_name}' already exists in {output_dir}. Skipping download.")
        return

    catalog = _find_event_catalog(event_name)
    if catalog is None:
        typer.echo(f"Event '{event_name}' not found in available catalogs.")
        raise typer.Exit(code=1)

    if catalog not in pe_zenodo_releases:
        typer.echo(f"No Zenodo release mapping found for catalog '{catalog}'.")
        return

    record_ids = _pe_zenodo_record_ids(catalog)
    pattern = _pe_glob_pattern(catalog, event_name)
    typer.echo(f"Queuing Zenodo download for {event_name}...")

    loop = asyncio.get_running_loop()
    for record_id in record_ids:
        download_func = partial(
            zenodo_download, record_id, output_dir=".", file_glob=pattern,
        )
        await loop.run_in_executor(None, download_func)
        if any(fname.endswith(('.hdf5', '.h5')) for fname in os.listdir(output_dir)):
            typer.echo(f"Finished downloading PE data for {event_name}")
            return

    typer.echo(f"Could not find PE data for '{event_name}' in Zenodo for catalog '{catalog}'.")

def _get_lvk_pe_data_filename(event_name):
    output_dir = f"{os.getcwd()}/{event_name}/official_pe"
    _ensure_dir(output_dir)
    if any(fname.endswith('.hdf5') for fname in os.listdir(output_dir)):
        return os.path.join(output_dir, next(fname for fname in os.listdir(output_dir) if fname.endswith('.hdf5')))
    else:
        _get_lvk_pe_data(event_name)
        return _get_lvk_pe_data_filename(event_name)  # Try again after downloading
def _get_lvk_pe_data(event_name):
    output_dir = f"{os.getcwd()}/{event_name}/official_pe"
    _ensure_dir(output_dir)
    if any(fname.endswith('.hdf5') for fname in os.listdir(output_dir)):
        typer.echo(f"PE data for '{event_name}' already exists in {output_dir}. Skipping download.")
        return

    catalog = _find_event_catalog(event_name)
    if catalog is None:
        typer.echo(f"Event '{event_name}' not found in available catalogs.")
        raise typer.Exit(code=1)

    if catalog not in pe_zenodo_releases:
        typer.echo(f"No Zenodo release mapping found for catalog '{catalog}'.")
        return

    record_ids = _pe_zenodo_record_ids(catalog)
    pattern = _pe_glob_pattern(catalog, event_name)
    for record_id in record_ids:
        typer.echo(f"Downloading PE data for '{event_name}' from Zenodo record {record_id}...")
        zenodo_download(record_id, output_dir=output_dir, file_glob=pattern)
        if any(fname.endswith(('.hdf5', '.h5')) for fname in os.listdir(output_dir)):
            typer.echo(f"Download complete. Files saved to: {output_dir}")
            return
    typer.echo(f"Could not find PE data for '{event_name}' in Zenodo for catalog '{catalog}'.")

def remove_duplicates(seq):
    # Each event has multiple versions, so the last bit in the name of '-vX'. We remove it and then remove duplicates.
    seen = set()
    result = []
    for item in seq:
        item_base = item.rsplit('-v', 1)[0]
        if item_base not in seen:
            seen.add(item_base)
            result.append(item_base)
    return result


def _list_lvk_data():
    """ List available data files. """
    _ensure_lvk_catalog_cache()
    return _lvk_events_cache


def check_event_name(event_name):
    events = _list_lvk_data()
    if event_name not in events:
        typer.echo(f"Event '{event_name}' not found in available events.")
        raise typer.Exit(code=1)
    return events


def list_data_lvk():
    """ List available data files. """
    events = _list_lvk_data()
    for event in events:
        typer.echo(event)
    typer.echo(f"\nTotal unique events: {len(events)}")
    return events


def _get_lvk_info_individual(event_name):
    ''' Download info data for a specific event and save in the same folder as the strain data. '''
    _ensure_dir(event_name)
    filename = f"{event_name}/{event_name}_info.json"
    if os.path.exists(filename):
        typer.echo(f"File {filename} already exists.")
        with open(filename, 'r') as f:
            info = json.load(f)
            info['detectors'] = set(info['detectors'])
        typer.echo(f"Data loaded from {filename}.")
        return info
    check_event_name(event_name)
    gps = event_gps(event_name)
    detectors = event_detectors(event_name)
    info = {
        'event_name': event_name,
        'gps': gps,
        'detectors': list(detectors)
    }
    with open(filename, 'w') as f:
        json.dump(info, f, indent=4)
    typer.echo(f"Saved event info to {filename}.")
    return info


def _strain_gwf_path(event_name, det, segment_length=60*20):
    if segment_length == 60*20:
        return f"{event_name}/{event_name}_{det}_strain.gwf"
    return f"{event_name}/{event_name}_{det}_strain_{segment_length//60}min.gwf"


def _fetch_detector_strain(det, start, end, filename):
    typer.echo(f"Fetching {det} data from {start} to {end}...")
    try:
        ts = gwpy.timeseries.TimeSeries.fetch_open_data(
            det, start, end, cache=True, sample_rate=4096,
        )
        ts.channel = f"{det}:GWOSC-STRAIN"
        ts.write(filename, format='gwf')
        typer.echo(f"Saved strain data to {filename} with channel {ts.channel}.")
        return ts
    except Exception as e:
        typer.echo(f"Error fetching data for {det}: {e}")
        return None


def _load_lvk_strain(event_name, return_data=True, download_pe=False, segment_length=60*20, info=None):
    """Load strain time series for an event from disk cache or GWOSC."""
    if info is None:
        info = _get_lvk_info_individual(event_name)
    gps = info['gps']
    detectors = info['detectors']
    typer.echo(f"Event '{event_name}' found with GPS time {gps}.")
    start, end = int(gps - segment_length/2), int(gps + segment_length/2)
    data = {}
    _ensure_dir(event_name)
    typer.echo(f"Downloading strain data for detectors: {', '.join(detectors)}")

    cached_paths = {}
    missing = []
    for det in detectors:
        filename = _strain_gwf_path(event_name, det, segment_length)
        if os.path.exists(filename):
            typer.echo(f"File {filename} already exists.")
            if return_data:
                cached_paths[det] = filename
        else:
            missing.append(det)

    if cached_paths and return_data:
        data.update(_read_gwfs(cached_paths))
        for filename in cached_paths.values():
            typer.echo(f"Data loaded from {filename}.")

    if missing:
        if len(missing) == 1:
            det = missing[0]
            filename = _strain_gwf_path(event_name, det, segment_length)
            fetched = _fetch_detector_strain(det, start, end, filename)
            if return_data and fetched is not None:
                data[det] = fetched
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(missing)) as executor:
                futures = {
                    det: executor.submit(
                        _fetch_detector_strain,
                        det,
                        start,
                        end,
                        _strain_gwf_path(event_name, det, segment_length),
                    )
                    for det in missing
                }
                for det, future in futures.items():
                    fetched = future.result()
                    if return_data and fetched is not None:
                        data[det] = fetched

    if download_pe:
        _get_lvk_pe_data(event_name)
    return data if return_data else None


async def _get_lvk_strain_individual(event_name, return_data=True, download_pe=False, segment_length=60*20):
    ''' Download strain data for a specific event. '''
    return _load_lvk_strain(
        event_name,
        return_data=return_data,
        download_pe=download_pe,
        segment_length=segment_length,
    )


def get_lvk_strain_individual_sync(event, download_pe=False, segment_length=60*20):
    return _load_lvk_strain(event, download_pe=download_pe, segment_length=segment_length)


def _get_lvk_strain_all_sync(download_pe=False, segment_length=60*20):
    events = _list_lvk_data()
    if download_pe:
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
            load = partial(_load_lvk_strain, download_pe=download_pe, segment_length=segment_length)
            results = list(executor.map(load, events))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(events) or 1)) as executor:
            load = partial(_load_lvk_strain, download_pe=False, segment_length=segment_length)
            results = list(executor.map(load, events))
    return events, dict(zip(events, results))


async def _get_lvk_strain_all(download_pe=False, segment_length=60*20):
    return _get_lvk_strain_all_sync(download_pe=download_pe, segment_length=segment_length)


def get_lvk_strain(event_name, download_pe, segment_length=60*20):
    if event_name == 'all':
        return _get_lvk_strain_all_sync(download_pe=download_pe, segment_length=segment_length)
    return event_name, _load_lvk_strain(event_name, download_pe=download_pe, segment_length=segment_length)


def get_pe(event_name):
    events = check_event_name(event_name)
    datasets = find_datasets(type='pe', event_name=event_name)


def h5_to_dict(h5_obj):
    result = {}
    for key, item in h5_obj.items():
        if isinstance(item, h5py.Dataset):
            result[key] = item[()]
        elif isinstance(item, h5py.Group):
            result[key] = h5_to_dict(item)
    return result


def _pe_file_path(event_name):
    output_dir = f"{os.getcwd()}/{event_name}/official_pe"
    _ensure_dir(output_dir)
    pe_files = [
        fname for fname in os.listdir(output_dir)
        if fname.endswith('.hdf5') or fname.endswith('.h5')
    ]
    if pe_files:
        return os.path.join(output_dir, pe_files[0])
    return None


def _approximant_group(pe_file):
    approximants = ['NRSur7dq4', 'SEOBNRv4PHM', 'IMRPhenomXPHM', 'IMRPhenomPv2', 'IMRPhenomD']
    approximants = [f"C00:{approx}" for approx in approximants] + approximants
    with h5py.File(pe_file, 'r') as f:
        for approx in approximants:
            if approx in f:
                return approx
        typer.echo(
            f"WARNING: No known approximant found in PE file. Available approximants: {list(f.keys())}"
        )
    return None


def _load_pe_psds(event_name):
    output_dir = f"{os.getcwd()}/{event_name}/official_pe"
    if not os.path.isdir(output_dir) or not any(
        fname.endswith(('.hdf5', '.h5')) for fname in os.listdir(output_dir)
    ):
        _get_lvk_pe_data(event_name)
    pe_path = _pe_file_path(event_name)
    if pe_path is None:
        typer.echo(f"No PE files found for event '{event_name}' in {output_dir}.")
        return None
    typer.echo(f"pe_path: {pe_path}")
    approx = _approximant_group(pe_path)
    if approx is None:
        return None
    with h5py.File(pe_path, 'r') as f:
        psds = h5_to_dict(f[approx]['psds'])
    typer.echo(f"Loaded PE PSDs from {pe_path} using approximant {approx}.")
    return psds


def _load_pe_samples(event_name):
    output_dir = f"{os.getcwd()}/{event_name}/official_pe"
    if not os.path.isdir(output_dir) or not any(
        fname.endswith(('.hdf5', '.h5')) for fname in os.listdir(output_dir)
    ):
        _get_lvk_pe_data(event_name)
    pe_path = _pe_file_path(event_name)
    if pe_path is None:
        typer.echo(f"No PE files found for event '{event_name}' in {output_dir}.")
        return None
    typer.echo(f"pe_path: {pe_path}")
    approx = _approximant_group(pe_path)
    if approx is None:
        return None
    with h5py.File(pe_path, 'r') as f:
        pe_samples = h5_to_dict(f[approx])
        typer.echo(f"Loaded PE samples from {pe_path} using approximant {approx}.")
    return pe_samples


def _load_noise_from_disk(event_name, detectors):
    paths = {}
    for det in detectors:
        before = f"{event_name}/{event_name}_{det}_noise_before.gwf"
        after = f"{event_name}/{event_name}_{det}_noise_after.gwf"
        if os.path.exists(before) and os.path.exists(after):
            paths[(det, 'before')] = before
            paths[(det, 'after')] = after
    if not paths:
        return None
    loaded = _read_gwfs(paths)
    noise = {}
    for det in detectors:
        before_key = (det, 'before')
        after_key = (det, 'after')
        if before_key in loaded and after_key in loaded:
            noise[det] = {'before': loaded[before_key], 'after': loaded[after_key]}
            typer.echo(
                f"Loaded cropped noise data for {det} from "
                f"{paths[before_key]} and {paths[after_key]}."
            )
    if len(noise) == len(detectors):
        return noise
    return None


def _crop_noise_around_signal(event_name, croplength=4, info=None, strain_data=None):
    """ Crop away the signal with +- `croplength` seconds around `gpstime`. For short BBH signals, 2 seconds is usually enough, but for low-mass signals you should use 16 seconds or more.
    `strain_data` is an optional dict {ifo: timeseries} to avoid re-reading strain GWF files.
    Returns: noise[ifo]['before'/'after']
    """
    if info is None:
        info = _get_lvk_info_individual(event_name)
    gpstime = info['gps']
    noise = _load_noise_from_disk(event_name, info['detectors'])
    if noise is not None:
        return noise
    if strain_data is None:
        strain_data = _load_lvk_strain(event_name, return_data=True, download_pe=False, info=info)
    noise = {}
    for det, ts in strain_data.items():
        t0 = ts.times.value[0]
        tf = ts.times.value[-1]
        noise[det] = {
            'before': ts.crop(t0, gpstime - croplength),
            'after': ts.crop(gpstime + croplength, tf),
        }
    for det in noise:
        noise_before_path = f"{event_name}/{event_name}_{det}_noise_before.gwf"
        noise_after_path = f"{event_name}/{event_name}_{det}_noise_after.gwf"
        noise[det]['before'].write(noise_before_path, format='gwf')
        noise[det]['after'].write(noise_after_path, format='gwf')
        typer.echo(f"Saved cropped noise data for {det} to {noise_before_path} and {noise_after_path}.")
    return noise

def _process_timeseries(event_name, info=None, strain_data=None):
    if info is None:
        info = _get_lvk_info_individual(event_name)
    if strain_data is None:
        strain_data = _load_lvk_strain(event_name, return_data=True, download_pe=False, info=info)
    from . import plots
    fig, ax = plots.plot_strain(strain_data)
    plots.save_figure(fig, f"{event_name}/{event_name}_strain.pdf")
    return None


def _process_psd_official(event_name, info=None, psds=None):
    if info is None:
        info = _get_lvk_info_individual(event_name)
    detectors = info['detectors']
    if psds is None:
        psds = _load_pe_psds(event_name)
    if psds is None:
        typer.echo(f"Cannot process PSD for event '{event_name}' because PE samples could not be loaded.")
        return None
    for det in detectors:
        if det in psds:
            f, psd = np.transpose(psds[det])
            np.save(f"{event_name}/{event_name}_{det}_psd.npy", [f, psd])
            typer.echo(f"Saved PSD for {det} to {event_name}/{event_name}_{det}_psd.npy")
        else:
            typer.echo(f"WARNING: No PSD found for detector '{det}' in PE samples for event '{event_name}'. Available detectors in PE samples: {list(psds.keys())}")
    from . import plots
    fig, ax = plots.plot_psd(psds)
    plots.save_figure(fig, f"{event_name}/{event_name}_psd.pdf")
    return psds

def _compute_welch_psd(ts):
    psd = ts.psd(fftlength=4, overlap=2)
    return psd.frequencies.value, psd.value


def _compute_welch_results(noise, detectors):
    tasks = [
        (det, side, noise[det][side])
        for det in detectors if det in noise
        for side in ['before', 'after']
    ]
    if len(tasks) <= 1:
        return {(det, side): _compute_welch_psd(ts) for det, side, ts in tasks}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {
            (det, side): executor.submit(_compute_welch_psd, ts)
            for det, side, ts in tasks
        }
        return {key: future.result() for key, future in futures.items()}


def _plot_psd_welch(event_name, info, noise, welch_results, official_psds=None):
    detectors = info['detectors']
    psds = {}
    for det in detectors:
        if det in noise:
            psds[det] = {}
            for side in ['before', 'after']:
                freqs, values = welch_results[(det, side)]
                psds[det][side] = np.transpose([freqs, values])
                np.save(f"{event_name}/{event_name}_{det}_{side}_psd_welch_seglen_4s.npy", [freqs, values])
                typer.echo(f"Saved Welch PSD for {det} ({side}) to {event_name}/{event_name}_{det}_{side}_psd_welch_seglen_4s.npy")
        else:
            typer.echo(f"WARNING: No strain noise found for detector '{det}' to compute Welch PSD for event '{event_name}'. Available detectors with strain noise: {list(noise.keys())}")
    psds_before = {det: psds[det]['before'] for det in psds}
    psds_after = {det: psds[det]['after'] for det in psds}
    from . import plots
    fig, ax = plots.plot_psd(psds_before)
    fig, ax = plots.plot_psd(psds_after, fig=fig)
    if official_psds is not None:
        fig, ax = plots.plot_psd(official_psds, fig=fig)
        for ax_i in ax:
            line = ax_i.get_lines()[-1]
            line.set_color('black')
    plots.save_figure(fig, f"{event_name}/{event_name}_psd_welch.pdf")
    return psds


def _process_psd_welch(event_name, info=None, noise=None, official_psds=None, welch_results=None):
    if info is None:
        info = _get_lvk_info_individual(event_name)
    if noise is None:
        noise = _crop_noise_around_signal(event_name, info=info)
    if welch_results is None:
        welch_results = _compute_welch_results(noise, info['detectors'])
    return _plot_psd_welch(event_name, info, noise, welch_results, official_psds=official_psds)


def process_lvk_event(event_name):
    info = _get_lvk_info_individual(event_name)
    loaded = _read_gwfs(_process_gwf_paths(event_name, info))
    strain_data, noise = _split_process_data(loaded, info['detectors'])
    if len(noise) < len(info['detectors']):
        noise = _crop_noise_around_signal(event_name, info=info, strain_data=strain_data)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        welch_future = executor.submit(_compute_welch_results, noise, info['detectors'])
        _process_timeseries(event_name, info=info, strain_data=strain_data)
        official_psds = _process_psd_official(event_name, info=info)
        welch_results = welch_future.result()
    _plot_psd_welch(event_name, info, noise, welch_results, official_psds=official_psds)
    return None

if __name__ == "__main__":
    event_name = "GW231123_135430"
    process_lvk_event(event_name)




