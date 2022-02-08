import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat

from brainio import CATALOG_NAME, BUCKET_NAME
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set, package_data_assembly_extend, package_data_assembly_commit

STIMULUS_SET_IDENTIFIER = 'object2vec'
NEUROID_ASSEMBLY_IDENTIFIER = 'object2vec'

N_SUBJECTS = 4
ROIS = ('EVC', 'LOC', 'PFS', 'OPA', 'PPA', 'RSC', 'FFA', 'OFA', 'STS', 'EBA')


def package_assembly(root_path: Path):
    for i_subject in range(N_SUBJECTS):
        subject_data_path = root_path / 'fmri' / 'subj{:03}'.format(i_subject + 1)
        roistack = loadmat(subject_data_path / 'roistack.mat')['roistack']
        roi_names = [d[0] for d in roistack['rois'][0, 0][:, 0]]
        conditions = [d[0] for d in roistack['conds'][0, 0][:, 0]]

        roi_indices = roistack['indices'][0, 0][0] - 1  # 1-indexed
        cv_groups = loadmat(subject_data_path / 'sets.mat')['sets']
        cv_groups = [[condition[0] for condition in cv_group[:, 0]] for cv_group in cv_groups[0, :]]
        cv_groups

        betas = roistack['betas'][0, 0]
        assembly = xr.DataArray(
            data=np.expand_dims(betas, axis=2),
            dims=('presentation', 'neuroid', 'time_bin'),
            coords={
                'image_id': ('presentation', conditions),
                'cv_group': ('presentation', [i_group for condition in conditions for i_group, group in enumerate(cv_groups) if condition in group]),
                'subject': ('neuroid', i_subject * np.ones(betas.shape[1])),
                'roi': ('neuroid', [roi_names[index] for index in roi_indices]),
                'time_bin_start': ('time_bin', [None]),
                'time_bin_end': ('time_bin', [None])
            },
        )

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


def package_stimuli(root_path: Path):
    image_paths = sorted((root_path / 'stimuli').rglob('*.png'))
    category_labels = [str(image_path.parent.name) for image_path in image_paths]
    metadata = pd.DataFrame({
        'image_id': np.arange(len(image_paths)),
        'category': category_labels,
    })
    stimulus_set = StimulusSet(metadata)
    stimulus_set.get_image = lambda image_id: image_paths[image_id]

    package_stimulus_set(
        catalog_name=CATALOG_NAME,
        proto_stimulus_set=stimulus_set,
        stimulus_set_identifier=STIMULUS_SET_IDENTIFIER,
        bucket_name=BUCKET_NAME,
    )


if __name__ == '__main__':
    root_path = Path(os.getenv('SHARED_DATASETS')) / 'object2vec'
    package_stimuli(root_path)
    package_assembly(root_path)

    # TODO discuss how to fix implementation: subjects view 810 images, but data only has 81 classes, since it's a block design (image_id in DataAssembly is not 1-1 with StimulusSet)
