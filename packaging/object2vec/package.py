import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat

from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set, package_data_assembly

STIMULUS_SET_IDENTIFIER = 'object2vec-stimuli'
NEUROID_ASSEMBLY_IDENTIFIER = 'object2vec'

N_SUBJECTS = 4
ROIS = ('EVC', 'LOC', 'PFS', 'OPA', 'PPA', 'RSC', 'FFA', 'OFA', 'STS', 'EBA')


def package_assembly(root_path: Path, catalog_name: str, bucket_name: str, stimulus_set: StimulusSet):
    subj_assemblies = []
    for i_subject in range(N_SUBJECTS):
        subject_data_path = root_path / 'fmri' / 'subj{:03}'.format(i_subject + 1)
        roistack = loadmat(subject_data_path / 'roistack.mat')['roistack']
        roi_names = [d[0] for d in roistack['rois'][0, 0][:, 0]]
        conditions = [d[0] for d in roistack['conds'][0, 0][:, 0]]

        roi_indices = roistack['indices'][0, 0][0] - 1  # 1-indexed
        cv_groups = loadmat(subject_data_path / 'sets.mat')['sets']
        cv_groups = [[condition[0] for condition in cv_group[:, 0]] for cv_group in cv_groups[0, :]]

        betas = roistack['betas'][0, 0]
        assembly = xr.DataArray(
            data=np.expand_dims(betas, axis=2),
            dims=('presentation', 'neuroid', 'time_bin'),
            coords={
                'image_id': ('presentation', np.arange(0, 810, 10)),
                f'cv_group_{i_subject}': ('presentation', [i_group for condition in conditions
                                                           for i_group, group in enumerate(cv_groups) if condition in group]),
                'subject': ('neuroid', i_subject * np.ones(betas.shape[1])),
                'roi': ('neuroid', [roi_names[index] for index in roi_indices]),
                'time_bin_start': ('time_bin', [None]),
                'time_bin_end': ('time_bin', [None])
            },
        )

        subj_assemblies.append(assembly)

    subj_assemblies = xr.concat(subj_assemblies, dim='neuroid')
    subj_assemblies.attrs['stimulus_set'] = stimulus_set

    package_data_assembly(
        proto_data_assembly=subj_assemblies,
        catalog_name=catalog_name,
        assembly_identifier=NEUROID_ASSEMBLY_IDENTIFIER,
        stimulus_set_identifier=STIMULUS_SET_IDENTIFIER,
        assembly_class='NeuronRecordingAssembly', 
        bucket_name=bucket_name,
    )


def package_stimuli(root_path: Path, catalog_name: str, bucket_name: str):
    image_paths = sorted((root_path / 'stimuli').rglob('*.png'))
    image_paths = {i: path for i, path in enumerate(image_paths)}
    category_labels = [str(image_path.parent.name) for image_path in image_paths.values()]
    metadata = pd.DataFrame({
        'image_id': list(image_paths.keys()),
        'category': category_labels,
    })
    stimulus_set = StimulusSet(metadata)
    stimulus_set.image_paths = image_paths

    package_stimulus_set(
        catalog_name=catalog_name,
        proto_stimulus_set=stimulus_set,
        stimulus_set_identifier=STIMULUS_SET_IDENTIFIER,
        bucket_name=bucket_name,
    )

    return stimulus_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Package the NSD dataset as a NeuroidAssembly and StimulusSet'
                                                 ' and upload to S3')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory containing stimuli and fMRI data')
    parser.add_argument('--bucket_name', type=str, default='bonnerlab-brainscore',
                        help='S3 bucket')
    parser.add_argument('--catalog_name', type=str, default='bonnerlab-brainscore',
                        help='brainio catalog')
    args = parser.parse_args()

    root_path = Path(os.getenv('SHARED_DATASETS')) / 'object2vec'
    stimulus_set = package_stimuli(root_path, catalog_name=args.catalog_name, bucket_name=args.bucket_name)
    package_assembly(root_path, catalog_name=args.catalog_name, bucket_name=args.bucket_name, stimulus_set)
