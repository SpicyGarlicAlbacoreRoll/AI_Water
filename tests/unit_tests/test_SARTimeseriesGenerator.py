import pytest
from src.SARTimeseriesGenerator import SARTimeseriesGenerator

@pytest.fixture
def supply_SARTimeseriesGenerator():
    sub_dataset_prefix = "test_subdataset"
    key_frame = "ulx_256_uly_256"
    dataset_name = "test_dataset"

    test_timeseries_frame_keys = [(sub_dataset_prefix, key_frame), (sub_dataset_prefix, key_frame)]
    test_timeseries_metadata = {
        sub_dataset_prefix: [
            key_frame: [
                [f"test/{sub_dataset_prefix}/S1A_test_file_0_vh_ulx_256_uly_256.tif", f"test/{sub_dataset_prefix}/S1A_test_file_0_vv_ulx_256_uly_256.tif"], 
                [f"test/{sub_dataset_prefix}/S1A_test_file_1_vh_ulx_256_uly_256.tif", f"test/{sub_dataset_prefix}/S1A_test_file_1_vv_ulx_256_uly_256.tif"]
            ]
        ]
    }

    return SARTimeseriesGenerator(
        test_timeseries_metadata, 
        time_series_frames=test_timeseries_frame_keys,
        batch_size=1,
        dim=(256, 256),
        time_steps=2,
        n_channels=2,
        output_dim=(256, 256),
        output_channels=1,
        n_classes=2,
        dataset_directory=dataset_dir(dataset),
        shuffle=True,
        min_samples=10)