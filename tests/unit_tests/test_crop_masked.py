import pytest
from src.dataset.crop_masked import open_dataset_metadata_file

# @pytest.fixture
# def supply_crop_masked_data():
#     return None
    # sub_dataset_prefix = "test_subdataset"
    # key_frame = "ulx_512_uly_512"
    # dataset_name = "test_dataset"

    # return CDLFrameData(sub_dataset_prefix, key_frame, dataset_name)

def test_open_dataset_metadata_file_non_json_input():
    data, key = open_dataset_metadata_file("dummy_file.txt", "dummy_dataset")
    res = (data, key)

    assert res == (None, None)

def test_open_dataset_metadata_file_non():
    dummy_data = {
    "test": {"ulx_512_uly_256": []},
    "train": {"ulx_0_uly_0": []}
}
    data, key = open_dataset_metadata_file("dummy_file.json", "dummy_dataset")
    res = (data, key)

    assert res == (dummy_data, "dummy_file")