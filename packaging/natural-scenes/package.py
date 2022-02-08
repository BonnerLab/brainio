from email.mime import image
from typing import List
import os
from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import h5py
from PIL import Image
import nibabel as nib
from tqdm import tqdm

from brainio import CATALOG_NAME, BUCKET_NAME
from brainio.stimuli import StimulusSet
from brainio.assemblies import NeuroidAssembly
from brainio.packaging import package_stimulus_set, package_data_assembly_extend, package_data_assembly_commit

NEUROID_ASSEMBLY_IDENTIFIER = 'allen2021.natural-scenes'
STIMULUS_SET_IDENTIFIER = 'allen2021.natural-scenes'
N_SUBJECTS = 8
N_MAX_SESSIONS = 40
N_TRIALS_PER_SESSION = 750
N_STIMULI = 73000
ROIS = {
    # 'streams': ('early', 'midventral', 'midlateral', 'midparietal', 'ventral', 'lateral', 'parietal'),
    'prf-visualrois': ('V1d', 'V1v', 'V2d', 'V2v', 'V3d', 'V3v', 'hV4'),
    # 'prf-eccrois': ('ecc0pt5', 'ecc1', 'ecc2', 'ecc4', 'ecc4+'),
    'floc-places': ('OPA', 'PPA', 'RSC'),
    'floc-faces': ('OFA', 'FFA-1', 'FFA-2', 'mTL-faces', 'aTL-faces'),
    'floc-bodies': ('EBA', 'FBA-1', 'FBA-2', 'mTL-bodies'),
    'floc-words': ('OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'mTL-words')
}


def extract_hdf5_images(root_path: Path):
    """
    converts HDF5-format NSD stimuli into BrainScore-compatible PNG files
    """
    if (root_path / 'images').exists():
        raise RuntimeError(f'directory f{root_path}/images already exists; delete it to continue')
    image_paths = [root_path / 'images' / f'{i_stimulus}.png' for i_stimulus in range(N_STIMULI)]
    stimuli = h5py.File(root_path / 'nsddata_stimuli' / 'stimuli' / 'nsd' / 'nsd_stimuli.hdf5', 'r')['imgBrick']
    for i_stimulus in range(N_STIMULI):
        image = Image.fromarray(stimuli[i_stimulus, :, :, :])
        image.save(image_paths[i_stimulus])
    return image_paths


def load_metadata(root_path: Path):
    metadata = pd.read_csv(root_path / 'nsddata' / 'experiments' / 'nsd' / 'nsd_stim_info_merged.csv', sep=',')
    metadata.rename(columns={'Unnamed: 0': 'image_id'}, inplace=True)


def package_stimuli(root_path: Path):
    image_paths = extract_hdf5_images(root_path)
    metadata = load_metadata(root_path)
    stimulus_set = StimulusSet(metadata)
    stimulus_set.get_image = lambda image_id: image_paths[int(image_id)]

    package_stimulus_set(
        catalog_name=CATALOG_NAME,
        proto_stimulus_set=stimulus_set,
        stimulus_set_identifier=STIMULUS_SET_IDENTIFIER,
        bucket_name=BUCKET_NAME,
    )


def format_id(idx: int) -> str:
    return f'{idx + 1:02}'  # subjects and sessions are 1-indexed


def extract_trial_info(root_path: Path):
    metadata = load_metadata(root_path)
    metadata = np.array(metadata.iloc[:, 17:])
    indices = np.nonzero(metadata)
    trials = metadata[indices[0], indices[1]] - 1  # convert from 1-indexing to 0-indexing

    image_ids = indices[0]
    subject_ids = indices[1] // 3  # each subject has 3 columns, 1 for each possible rep
    session_ids = trials // N_TRIALS_PER_SESSION
    intra_session_trial_ids = trials % N_TRIALS_PER_SESSION

    trial_info = xr.DataArray(
        np.full((N_SUBJECTS, N_MAX_SESSIONS, N_TRIALS_PER_SESSION), np.nan, dtype=np.int64),
        dims=('subject', 'session', 'trial'),
        coords={
            'subject': np.arange(N_SUBJECTS),
            'session': np.arange(N_MAX_SESSIONS),
            'trial': np.arange(N_TRIALS_PER_SESSION),
        },
    )
    trial_info.values[subject_ids, session_ids, intra_session_trial_ids] = image_ids
    return trial_info


def format_roi_data(root_path: Path):
    roi_data = [[] for _ in range(N_SUBJECTS)]
    for i_subject in range(N_SUBJECTS):
        subject_id = format_id(i_subject)
        roi_path = root_path / 'nsddata' / 'ppdata' / ('subj' + subject_id) / 'func1pt8mm' / 'roi'
        for roi_type in ROIS.keys():
            label_path = root_path / 'nsddata' / 'freesurfer' / ('subj' + subject_id) / 'label' / (roi_type + '.mgz.ctab')

            roi_data[i_subject].append(pd.read_csv(label_path, delim_whitespace=True, names=('value', 'roi')))
            roi_data[i_subject][-1]['roi_type'] = [roi_type] * len(roi_data[i_subject][-1])

            roi_indices = []
            for _, row in roi_data[i_subject][-1].iterrows():
                roi_volume = nib.load(roi_path / (roi_type + '.nii.gz')).get_fdata()
                roi_indices.append(np.where(roi_volume == row['value']))
            roi_data[i_subject][-1]['indices'] = roi_indices

        roi_data[i_subject] = pd.concat(roi_data[i_subject], ignore_index=True)
        roi_data[i_subject] = roi_data[i_subject].set_index('roi')
    return roi_data


def load_ncsnr(i_subject: int):
    subject_id = format_id(i_subject)
    ncsnr_path = root_path / 'nsddata_betas' / 'ppdata' / ('subj' + subject_id) \
        / 'func1pt8mm' / 'betas_fithrf_GLMdenoise_RR' / 'ncsnr.nii.gz'
    ncsnr = nib.load(ncsnr_path).get_fdata()
    return ncsnr


def load_betas(root_path: Path, i_subject: int, i_session: int, z_score=True):
    session_id = format_id(i_session)
    subject_id = format_id(i_subject)
    session_path = root_path / 'nsddata_betas' / 'ppdata' / ('subj' + subject_id) \
        / 'func1pt8mm' / 'betas_fithrf_GLMdenoise_RR' / ('betas_session' + session_id + '.hdf5')
    betas = h5py.File(session_path, 'r')['betas']
    betas = np.array(betas, dtype=np.single) / 300  # converting to % signal change

    if z_score:
    # z-scoring betas across all brain_voxels
    # TODO check if this was what was recommended in the manual
        mean = np.nanmean(betas, axis=(1, 2, 3))
        std = np.nanstd(betas, axis=(1, 2, 3))
        betas = np.transpose((np.transpose(betas, axes=(1, 2, 3, 0)) - mean) / std, axes=(3, 0, 1, 2))
    return betas


def average_across_reps(assembly: NeuroidAssembly, return_groupby=False):
    groupby = assembly.groupby('image_id')
    assembly_grouped = groupby.mean()
    if return_groupby:
        return assembly_grouped, groupby
    else:
        return assembly_grouped


def compute_nc(assembly: NeuroidAssembly):
    assembly_grouped, groupby = average_across_reps(assembly, return_groupby=True)
    counts = np.array([len(reps) for reps in groupby.groups.values()])
    ncsnr = assembly_grouped['ncsnr'].values

    ncsnr_squared = ncsnr ** 2
    if counts is None:
        fraction = 1
    else:
        unique, counts = np.unique(counts, return_counts=True)
        reps = dict(zip(unique, counts))
        fraction = (reps[1] + reps[2] / 2 + reps[3] / 3) / (reps[1] + reps[2] + reps[3])
    return ncsnr_squared / (ncsnr_squared + fraction)


def package_assembly(root_path: Path):
    roi_data = format_roi_data(root_path)
    trial_info = extract_trial_info(root_path)

    for i_subject in tqdm(range(N_SUBJECTS), desc='subject'):
        subject_id = format_id(i_subject)
        betas_path = root_path / 'nsddata_betas' / 'ppdata' / ('subj' + subject_id) / 'func1pt8mm' / 'betas_fithrf_GLMdenoise_RR'

        sessions = betas_path.glob('betas_session*.hdf5')
        n_sessions = len(list(sessions))
        n_trials = n_sessions * N_TRIALS_PER_SESSION

        rois = list(itertools.chain(*[[roi for roi in rois] for rois in ROIS.values()]))

        roi_indices = [[] for _ in range(3)]
        roi_labels = list(itertools.chain(*[[roi] * len(roi_data[i_subject]['indices'][roi][0]) for roi in rois]))
        for dimension in range(3):
            for roi in rois:
                roi_indices[dimension].append(roi_data[i_subject]['indices'][roi][dimension])
            roi_indices[dimension] = np.concatenate(roi_indices[dimension])

        ncsnr = load_ncsnr(i_subject)
        ncsnr = ncsnr[roi_indices[0], roi_indices[1], roi_indices[2]]

        n_voxels = len(roi_labels)

        betas = np.empty((n_trials, n_voxels))
        session_ids = np.empty((n_trials,))
        trial_ids = np.empty((n_trials,))

        for i_session in tqdm(range(n_sessions), desc='session'):
            session_betas = load_betas(i_subject, i_session)
            session_betas = session_betas[:, roi_indices[0], roi_indices[1], roi_indices[2]]

            i_start = i_session * N_TRIALS_PER_SESSION
            i_end = i_start + N_TRIALS_PER_SESSION
            betas[i_start:i_end, :] = session_betas
            trial_ids[i_start:i_end] = np.arange(N_TRIALS_PER_SESSION)
            session_ids[i_start:i_end] = i_session

        image_ids = trial_info.values[i_subject, :n_sessions, :].flatten()

        assembly = NeuroidAssembly(
            np.expand_dims(betas, axis=2),  # BrainScore expects a time_bin dimension
            dims=('presentation', 'neuroid', 'time_bin'),
            coords={
                'image_id': ('presentation', image_ids),
                'session_id': ('presentation', session_ids),
                'trial_id': ('presentation', trial_ids),
                'subject': ('neuroid', (i_subject - 1) * np.ones(betas.shape[1])),
                'roi': ('neuroid', roi_labels),
                'x': ('neuroid', roi_indices[0]),
                'y': ('neuroid', roi_indices[1]),
                'z': ('neuroid', roi_indices[2]),
                'ncsnr': ('neuroid', ncsnr),
                'time_bin_start': ('time_bin', [None]),
                'time_bin_end': ('time_bin', [None])
            }
        )

        nc = compute_nc(assembly)
        assembly = assembly.assign_coords({'nc': ('neuroid', nc)})
        assembly = assembly.reset_index(list(assembly.indexes))  # TODO check if this is necessary or if package_* take care of it

        package_data_assembly_extend(
            assembly,
            extending_dim='neuroid',
            assembly_identifier=NEUROID_ASSEMBLY_IDENTIFIER,
            assembly_class='NeuronRecordingAssembly',
        )

    package_data_assembly_commit(
        catalog_name=CATALOG_NAME,
        assembly_identifier=NEUROID_ASSEMBLY_IDENTIFIER,
        stimulus_set_identifier=STIMULUS_SET_IDENTIFIER,
        assembly_class='NeuronRecordingAssembly', 
        bucket_name=BUCKET_NAME,
    )


if __name__ == '__main__':
    root_path = Path(os.getenv('SHARED_DATASETS')) / 'natural-scenes'
    package_stimuli(root_path)
    package_assembly(root_path)
