import pytest
from src.CDLFrameData import CDLFrameData


@pytest.fixture
def supply_CDLFrameData():
    sub_dataset_prefix = "test_subdataset"
    key_frame = "ulx_512_uly_512"
    dataset_name = "test_dataset"

    return CDLFrameData(sub_dataset_prefix, key_frame, dataset_name)

def test_get_sub_dataset_name(supply_CDLFrameData):
    sub_dataset_name = supply_CDLFrameData.get_sub_dataset_name()
    assert sub_dataset_name == "test_subdataset"

def test_get_frame_index_key(supply_CDLFrameData):
    frame_index_key = supply_CDLFrameData.get_frame_index_key()
    assert frame_index_key == "ulx_512_uly_512"


# Private member functions

def test_sort_data(supply_CDLFrameData):
    test_data = [("S1B_201203_VV.tif", "S1B_201203_VH.tif"), ("S1A_201202_VV.tif", "S1A_201202_VH.tif"), ("S1B_201201_VV.tif", "S1B_201201_VH.tif")]
    sorted_data = [("S1B_201201_VV.tif", "S1B_201201_VH.tif"), ("S1A_201202_VV.tif", "S1A_201202_VH.tif"), ("S1B_201203_VV.tif", "S1B_201203_VH.tif")]

    test_data = supply_CDLFrameData._CDLFrameData__sort_data(test_data)

    assert test_data == sorted_data

def test_random_frame_sample_longer(supply_CDLFrameData):
        # in the event that the timeseries sample has more timesteps than what what the user wants
        test_data_longer = [("S1B_201203_VV.tif", "S1B_201203_VH.tif"), ("S1A_201202_VV.tif", "S1A_201202_VH.tif"), ("S1B_201201_VV.tif", "S1B_201201_VH.tif")]
        time_steps = 2

        random_selection = supply_CDLFrameData._CDLFrameData__random_frame_sample(test_data_longer, time_steps)
        
        assert len(random_selection) == time_steps

def test_random_frame_sample_shorter(supply_CDLFrameData):      
        # in the event that the timeseries sample has fewer timesteps than what what the user wants
        test_data_shorter = [("S1B_201203_VV.tif", "S1B_201203_VH.tif")]
        time_steps = 2

        random_selection = supply_CDLFrameData._CDLFrameData__random_frame_sample(test_data_shorter, time_steps)
        
        assert len(random_selection) < time_steps

def test_extend_frame_sample(supply_CDLFrameData):
    #given a time series with fewer time steps than required, pad out the sequence with randomly selected repeats
    test_data = [("S1B_201203_VV.tif", "S1B_201203_VH.tif"), ("S1A_201202_VV.tif", "S1A_201202_VH.tif"), ("S1B_201201_VV.tif", "S1B_201201_VH.tif")]
    time_steps = 6

    output = supply_CDLFrameData._CDLFrameData__extend_sample_timesteps(test_data, time_steps)

    assert len([sample for sample in output if sample not in test_data]) == 0

def test_extend_frame_sample_length(supply_CDLFrameData):
    #given a time series with fewer time steps than required, pad out the sequence with randomly selected repeats
    test_data = [("S1B_201203_VV.tif", "S1B_201203_VH.tif"), ("S1A_201202_VV.tif", "S1A_201202_VH.tif"), ("S1B_201201_VV.tif", "S1B_201201_VH.tif")]
    time_steps = 6

    output = supply_CDLFrameData._CDLFrameData__extend_sample_timesteps(test_data, time_steps)
    assert len(output) > len(test_data)

