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


def test_decoding_to_float():
    b_arr = np.zeros(shape=(32,), dtype=np.uint8)
    b_arr[2: 7] = 1
    b_arr[10] = 1
    int_arr = np.packbits(b_arr, bitorder="big")
    f_val = FzdDataReader._decode_uint8_to_float32(int_arr=int_arr, encoding="big")
    assert f_val == 0.15625
