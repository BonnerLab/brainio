import logging
from pathlib import Path

import pandas as pd

from brainio.lookup import CATALOG_PATH_KEY

_logger = logging.getLogger(__name__)

def brainio_test():
    path = Path(__file__).parent / "lookup.csv"
    _logger.debug(f"Loading lookup from {path}")
    print(f"Loading lookup from {path}")  # print because logging usually isn't set up at this point during import
    df = pd.read_csv(path)
    df.attrs[CATALOG_PATH_KEY] = str(path)
    return df


def brainio_test2():
    path = Path(__file__).parent / "lookup2.csv"
    _logger.debug(f"Loading lookup from {path}")
    print(f"Loading lookup from {path}")  # print because logging usually isn't set up at this point during import
    df = pd.read_csv(path)
    df.attrs[CATALOG_PATH_KEY] = str(path)
    return df

