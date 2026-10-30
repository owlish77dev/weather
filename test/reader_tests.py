import pytest
from weather.reading import FzdDataReader, snow_path
import numpy as np


@pytest.fixture()
def read_fzd_data():
    fdr = FzdDataReader(files=snow_path)
    fdr.read_all_data()
    return fdr


def test_fzd_data_reading(read_fzd_data):
    fdr = FzdDataReader(files=snow_path)
    assert fdr is not None

