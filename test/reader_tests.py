import pytest
from weather.reading import FzdDataReader, snow_path


@pytest.fixture()
def read_fzd_data():
    fdr = FzdDataReader(files=snow_path.joinpath("2021-12-04-02-29-39_vis_10000_rain_0_synop_70_0.bag"))
    fdr.read_all_data()
    return fdr


@pytest.fixture()
def read_two_fzd_data():
    bags = [
        snow_path.joinpath("2021-12-04-02-29-39_vis_10000_rain_0_synop_70_0.bag"),
        snow_path.joinpath("2021-12-04-02-29-39_vis_10000_rain_0_synop_70_1.bag")
    ]
    fdr = FzdDataReader(files=bags)
    fdr.read_all_data()
    return fdr


def test_fzd_data_reading(read_fzd_data):
    """Read single bag file."""
    fdr = read_fzd_data
    assert fdr is not None
    assert fdr.weather.shape[0] > 0
    assert fdr.transformation.shape[0] > 0


def test_read_two_bag_files(read_two_fzd_data):
    """Read two bag files."""
    fdr = read_two_fzd_data
    assert fdr is not None
    assert fdr.weather.shape[0] > 0
    assert fdr.transformation.shape[0] > 0

