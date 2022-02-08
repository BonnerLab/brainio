import os
from pathlib import Path
import pickle

from brainio import CATALOG_NAME, BUCKET_NAME
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

STIMULUS_SET_IDENTIFIER = 'ade20k'


def load_metadata(root_path: Path):
    metadata_path = root_path / 'ADE20K_2021_17_01' / 'index_ade20k.pkl'
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return metadata


def package_stimuli(root_path: Path):
    metadata = load_metadata(root_path)
    metadata['image_id'] = 'filename'
    image_paths = {row['image_id']: Path(row['folder']) / row['filename'] for row in metadata.iterrows()}
    metadata = metadata.drop(['folder', 'filename'])

    stimulus_set = StimulusSet(metadata)
    stimulus_set.get_image = lambda image_id: image_paths[image_id]

    package_stimulus_set(
        catalog_name=CATALOG_NAME,
        proto_stimulus_set=stimulus_set,
        stimulus_set_identifier=STIMULUS_SET_IDENTIFIER,
        bucket_name=BUCKET_NAME,
    )


if __name__ == '__main__':
    root_path = os.getenv('SHARED_DATASETS') / 'ade20k'
    package_stimuli(root_path)
