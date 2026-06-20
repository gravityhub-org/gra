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

current_dir = os.getcwd()

pe_zenodo_releases = {}
pe_zenodo_releases['GWTC-2.1-confident'] = 'https://zenodo.org/records/6513631'
pe_zenodo_releases['GWTC-3-confident'] = 'https://zenodo.org/records/5546663'
pe_zenodo_releases['GWTC-4.0'] = 'https://zenodo.org/records/17014085'


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _read_gwf(filename):
    """Read a GWF file, inferring the channel name automatically."""
    channel = frtools.get_channels(filename)[0]
    return gwpy.timeseries.TimeSeries.read(filename, format='gwf', channel=channel)


def _find_event_catalog(event_name):
    """Return the catalog name that contains *event_name*, or None."""
    for cat in list(pe_zenodo_releases.keys()) + ['O4_Discovery_Papers']:
        if any(event_name in e for e in find_datasets(type='event', catalog=cat)):
            return cat
    return None


def _pe_glob_pattern(catalog, event_name):
    """Return the Zenodo file-glob pattern for a given catalog and event."""
    if catalog == "GWTC-4.0":
        return f"*{event_name}*"
    return f"*{event_name}*nocosmo*.h5"


async def _get_lvk_pe_data_async(event_name):
    output_dir = f"{current_dir}/{event_name}"
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

    record_id = pe_zenodo_releases[catalog].split('/')[-1]
    typer.echo(f"Queuing Zenodo download for {event_name}...")

    loop = asyncio.get_running_loop()
    download_func = partial(zenodo_download, record_id, output_dir=".", file_glob=_pe_glob_pattern(catalog, event_name))
    await loop.run_in_executor(None, download_func)

    typer.echo(f"Finished downloading PE data for {event_name}")

def _get_lvk_pe_data_filename(event_name):
    output_dir = f"{current_dir}/{event_name}/official_pe"
    _ensure_dir(output_dir)
    if any(fname.endswith('.hdf5') for fname in os.listdir(output_dir)):
        return os.path.join(output_dir, next(fname for fname in os.listdir(output_dir) if fname.endswith('.hdf5')))
    else:
        _get_lvk_pe_data(event_name)
        return _get_lvk_pe_data_filename(event_name)  # Try again after downloading
def _get_lvk_pe_data(event_name):
    output_dir = f"{current_dir}/{event_name}/official_pe"
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

    record_id = pe_zenodo_releases[catalog].split('/')[-1]
    typer.echo(f"Downloading PE data for '{event_name}' from Zenodo record {record_id}...")
    zenodo_download(record_id, output_dir=output_dir, file_glob=_pe_glob_pattern(catalog, event_name))
    typer.echo(f"Download complete. Files saved to: {output_dir}")

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
    catalogs = ['GWTC-1-confident',
                'GWTC-2.1-confident',
                'GWTC-3-confident',
                'GWTC-4.0',
                'O4_Discovery_Papers']
    events_all = []
    for catalog in catalogs:
        events = find_datasets(type='events', catalog=catalog)
        events = remove_duplicates(events)
        events_all.extend(events)
    return events_all


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


async def _get_lvk_strain_individual(event_name, return_data=True, download_pe=False, segment_length=60*20):
    ''' Download strain data for a specific event. '''
    info = _get_lvk_info_individual(event_name)
    gps = info['gps']
    detectors = info['detectors']
    typer.echo(f"Event '{event_name}' found with GPS time {gps}.")
    start, end = int(gps - segment_length/2), int(gps + segment_length/2)
    data = {}
    _ensure_dir(event_name)
    typer.echo(f"Downloading strain data for detectors: {', '.join(detectors)}")
    for det in detectors:
        if segment_length == 60*20:
            filename = f"{event_name}/{event_name}_{det}_strain.gwf"
        else:
            filename = f"{event_name}/{event_name}_{det}_strain_{segment_length//60}min.gwf"
        if os.path.exists(filename):
            typer.echo(f"File {filename} already exists.")
            if return_data:
                data[det] = _read_gwf(filename)
                typer.echo(f"Data loaded from {filename}.")
            continue
        typer.echo(f"Fetching {det} data from {start} to {end}...")
        try:
            data[det] = gwpy.timeseries.TimeSeries.fetch_open_data(det, start, end, cache=True, sample_rate=4096)
            data[det].channel = f"{det}:GWOSC-STRAIN"
            data[det].write(filename, format='gwf')
            typer.echo(f"Saved strain data to {filename} with channel {data[det].channel}.")
        except Exception as e:
            typer.echo(f"Error fetching data for {event_name} with {det}: {e}")
    if download_pe:
        _get_lvk_pe_data(event_name)
    return data if return_data else None


def get_lvk_strain_individual_sync(event, download_pe=False, segment_length=60*20):
    return asyncio.run(_get_lvk_strain_individual(event, download_pe=download_pe, segment_length=segment_length))


async def _get_lvk_strain_all(download_pe=False, segment_length=60*20):
    events = _list_lvk_data()
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:  # FIXME: The ProcessPoolExecutor is used instead of ThreadPoolExecutor because the latter runs into issues with the zenodo's download function, which is not async safe (e.g., uses global variables). A separate process for each download seems to work because it isolates the memory. However, it's not ideal.
        tasks = [loop.run_in_executor(executor, get_lvk_strain_individual_sync, event, download_pe, segment_length) for event in events]
        results = await asyncio.gather(*tasks)
    data_all = dict(zip(events, results))
    return events, data_all


def get_lvk_strain(event_name, download_pe, segment_length=60*20):
    if event_name == 'all':
        events, data_all = asyncio.run(_get_lvk_strain_all(download_pe=download_pe, segment_length=segment_length))
        return events, data_all
    else:
        data = asyncio.run(_get_lvk_strain_individual(event_name, download_pe=download_pe, segment_length=segment_length))
        return event_name, data


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


def _load_pe_samples(event_name):
    output_dir = f"{current_dir}/{event_name}/official_pe"
    if not os.path.isdir(output_dir) or not any(
        fname.endswith(('.hdf5', '.h5')) for fname in os.listdir(output_dir)
    ):
        _get_lvk_pe_data(event_name)
    if not os.path.isdir(output_dir):
        typer.echo(f"No PE directory for event '{event_name}' at {output_dir}.")
        return None
    pe_files = [fname for fname in os.listdir(output_dir) if fname.endswith('.hdf5') or fname.endswith('.h5')]
    if len(pe_files) == 0:
        typer.echo(f"No PE files found for event '{event_name}' in {output_dir}.")
        return None
    elif len(pe_files) > 1:
        typer.echo(f"Multiple PE files found for event '{event_name}' in {output_dir}. Using the first one: {pe_files[0]}")
    pe_file = pe_files[0]
    pe_path = os.path.join(output_dir, pe_file)
    typer.echo(f"pe_path: {pe_path}")
    with h5py.File(pe_path, 'r') as f:
        approximants = ['NRSur7dq4', 'SEOBNRv4PHM', 'IMRPhenomXPHM', 'IMRPhenomPv2', 'IMRPhenomD']
        approximants = [f"C00:{approx}" for approx in approximants] + approximants
        dataset = None
        for approx in approximants:
            if approx in f:
                dataset = f[approx]
                break
        if dataset is None:
            typer.echo(f"WARNING: No known approximant found in PE file for event '{event_name}'. Available approximants: {list(f.keys())}")
            return None
        pe_samples = h5_to_dict(dataset)
        typer.echo(f"Loaded PE samples from {pe_path} using approximant {approx}.")
    return pe_samples


def _crop_noise_around_signal(event_name, croplength=4):
    """ Crop away the signal with +- `croplength` seconds around `gpstime`. For short BBH signals, 2 seconds is usually enough, but for low-mass signals you should use 16 seconds or more.
    `data` is a dict {ifo: timeseries}.
    Returns: noise[ifo]['before'/'after']
    """
    info = _get_lvk_info_individual(event_name)
    gpstime = info['gps']
    noise = {}
    for det in info['detectors']:
        noise_before_path = f"{event_name}/{event_name}_{det}_noise_before.gwf"
        noise_after_path = f"{event_name}/{event_name}_{det}_noise_after.gwf"
        if os.path.exists(noise_before_path) and os.path.exists(noise_after_path):
            noise[det] = {
                'before': _read_gwf(noise_before_path),
                'after': _read_gwf(noise_after_path),
            }
            typer.echo(f"Loaded cropped noise data for {det} from {noise_before_path} and {noise_after_path}.")
    if len(noise) == len(info['detectors']):
        return noise
    data = asyncio.run(_get_lvk_strain_individual(event_name, return_data=True, download_pe=False))
    noise = {}
    for det, ts in data.items():
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

def _process_timeseries(event_name):
    info = _get_lvk_info_individual(event_name)
    data = asyncio.run(_get_lvk_strain_individual(event_name, return_data=True, download_pe=False))
    # Crop the signal out of the data and save the results
    gpstime = info['gps']
    noises = _crop_noise_around_signal(event_name)
    from . import plots
    fig, ax = plots.plot_strain(data)
    fig.savefig(f"{event_name}/{event_name}_strain.pdf", bbox_inches='tight')
    return None


def _process_psd_official(event_name):
    info = _get_lvk_info_individual(event_name)
    detectors = info['detectors']
    pe_samples = _load_pe_samples(event_name)
    if pe_samples is None:
        typer.echo(f"Cannot process PSD for event '{event_name}' because PE samples could not be loaded.")
        return None
    psds = pe_samples['psds']
    for det in detectors:
        if det in psds:
            f, psd = np.transpose(psds[det])
            np.save(f"{event_name}/{event_name}_{det}_psd.npy", [f, psd])
            typer.echo(f"Saved PSD for {det} to {event_name}/{event_name}_{det}_psd.npy")
        else:
            typer.echo(f"WARNING: No PSD found for detector '{det}' in PE samples for event '{event_name}'. Available detectors in PE samples: {list(psds.keys())}")
    from . import plots
    fig, ax = plots.plot_psd(psds)
    fig.savefig(f"{event_name}/{event_name}_psd.pdf", bbox_inches='tight')
    return psds

def _process_psd_welch(event_name):
    info = _get_lvk_info_individual(event_name)
    detectors = info['detectors']
    noise = _crop_noise_around_signal(event_name)
    psds = {}
    for det in detectors:
        if det in noise:
            psds[det] = {}
            for side in ['before', 'after']:
                psd = noise[det][side].psd(fftlength=4, overlap=2)
                freqs = psd.frequencies.value
                psds[det][side] = np.transpose([freqs, psd.value])
                np.save(f"{event_name}/{event_name}_{det}_{side}_psd_welch_seglen_4s.npy", [freqs, psd.value])
                typer.echo(f"Saved Welch PSD for {det} ({side}) to {event_name}/{event_name}_{det}_{side}_psd_welch_seglen_4s.npy")
        else:
            typer.echo(f"WARNING: No strain noise found for detector '{det}' to compute Welch PSD for event '{event_name}'. Available detectors with strain noise: {list(noise.keys())}")
    psds_before = {det: psds[det]['before'] for det in psds}
    psds_after = {det: psds[det]['after'] for det in psds}
    from . import plots
    fig, ax = plots.plot_psd(psds_before)
    fig, ax = plots.plot_psd(psds_after, fig=fig)
    # Load the official psd and plot it out:
    official_psds = _process_psd_official(event_name)
    if official_psds is not None:
        fig, ax = plots.plot_psd(official_psds, fig=fig)
        # Change the color of the 'official' psd to black:
        for ax_i in ax:
            line = ax_i.get_lines()[-1]  # Get the last line, which should be the official psd
            line.set_color('black')
    fig.savefig(f"{event_name}/{event_name}_psd_welch.pdf", bbox_inches='tight')
    return psds

def process_lvk_event(event_name):
    _process_timeseries(event_name)
    _process_psd_official(event_name)
    _process_psd_welch(event_name)
    return None

if __name__ == "__main__":
    event_name = "GW231123_135430"
    process_lvk_event(event_name)




