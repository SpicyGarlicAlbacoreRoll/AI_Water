"""
Train or test a neural network.

To train a network:
    '$ python3 main.py train name_of_net name_of_dataset --epochs 10'
To test a trained network:
    '$ python3 main.py test name_of_net name_of_dataset'
For more information see README.md
"""

import os
import json
from datetime import date, datetime 
from argparse import ArgumentParser, Namespace

from src.asf_cnn import test_model_masked, train_model, test_model_timeseries
from src.model import load_model, path_from_model_name
from src.model.architecture.masked import create_model_masked
from src.config import NETWORK_DEMS, BATCH_SIZE
from src.model.architecture.dice_loss import jaccard_distance_loss
from PIL import Image, ImageOps
from keras.optimizers import Adam
from keras.metrics import MeanIoU
from keras.losses import BinaryCrossentropy
from src.model.architecture.dice_loss import dice_coefficient_loss, dice_coefficient, cosh_dice_coefficient_loss
from src.plots import write_mask_to_file
from src.gdal_wrapper import gdal_open

from src.model.architecture.crop_masked import create_cdl_model_masked
from keras.preprocessing.image import array_to_img
import numpy as np

def train_wrapper(args: Namespace) -> None:
    """ Function for training a network. """
    model_name = args.model
    if args.cont:
        model = load_model(model_name)
        history = model.__asf_model_history
        weights = model.get_weights()
        # optimizer = model.optimizer
        model.compile(
            loss="mean_squared_error", optimizer=Adam(learning_rate=1e-3), metrics=['accuracy', MeanIoU(num_classes=2)]
        )
        model.set_weights(weights)
    #     model.compile(
    #         loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"]
    # )
    else:
        model_path = path_from_model_name(model_name)
        if not args.overwrite and os.path.isfile(model_path):
            print(f"File {model_name} already exists!")
            return

        # model = create_model_masked(model_name)
        model = create_cdl_model_masked(model_name)
        history = {'loss': [],  'accuracy': [], "mean_io_u": []}

    train_model(model, history, args.dataset, args.epochs)


def test_wrapper(args: Namespace) -> None:
    model_name = args.model
    model = load_model(model_name)

    # if args.edit:
    #     predictions, data_iter, metadata = test_model_masked(
    #         model, args.dataset, args.edit
    #     )
    #     edit_predictions(
    #         predictions, data_iter, metadata
    #     )
    # else:
    predictions, test_batch_metadata, input_metadata = test_model_timeseries(
        model, args.dataset, args.edit
    )

    model_batch_size = len(test_batch_metadata[0])
    current_date_time = str(datetime.utcnow())

    # Version number _ Country _ Region
    model_name_metadata = model_name.split("_")
    
    metadata = {"model_test_info": {"name": model_name, "model_architecture_version": model_name_metadata[0], "dataset": args.dataset, "batch_size": model_batch_size, "UTC_date_time": current_date_time}}

    prediction_directory_name = "{0}_{1}_{2}".format(model_name, args.dataset, current_date_time)

    os.mkdir("predictions/{0}".format(prediction_directory_name))

    # {
       # model_info: {}
       # batch_x [
       #    [
       #    {
       #    sample_x: {
       #            timesteps: [
       #                [
       #                    test/WA_2018/S1A_ulx_0000_uly_0000.tif    
       #                ]
       #            ]
       #        }
       # }
       #    ]
       # ]
    # }

    # for batches
    non_blank_predictions = 0
    for idx, batch in enumerate(test_batch_metadata):
        metadata["batch_{0}".format(idx)] = []
        samples = []
        # for sample in batch
        for idy, sample in enumerate(batch):
            current_prediction_idx = idx * BATCH_SIZE + idy
            current_subdataset = sample[0][0].split("/")[1]    #IE: WA_2018, AK_2020
            prediction_subdataset_name = f"predictions/{prediction_directory_name}/{current_subdataset}"

            if not os.path.isdir(prediction_subdataset_name):
                os.mkdir(prediction_subdataset_name)

            if len(predictions) > current_prediction_idx:
                image = predictions[current_prediction_idx]
                image = np.array(image[:, :, 0].reshape(NETWORK_DEMS, NETWORK_DEMS, 1)).astype(dtype=np.uint8)

                if np.ptp(image) != 0:
                    # img_0 = array_to_img(image)
                    prediction_frame = "_".join(sample[0][0].split("_")[-4:])
                    filename = f"{prediction_subdataset_name}/CDL_{current_subdataset}_prediction_{prediction_frame}"
                    dataset_path_to_sample = f"datasets/{args.dataset}/{sample[0][0]}"
                    # filename_0 = "predictions/{0}/batch_{1}/batch_{1}_sample_{2}.tif".format(prediction_directory_name, batch_index, idy % model_batch_size)
                    save_img(filename, dataset_path_to_sample, image)
                    img_0 = array_to_img(image)
                    img_0.save(filename.replace(".tif", ".png"))
                    non_blank_predictions+=1


            timeseries_sample = {}
            timeseries_sample["timesteps"] = sample
            # The name of the prediction produced by this sample
            prediction_file_name=f"predictions/{prediction_directory_name}/batch_{idx}/_sample{idy}"
            sample_data = {f"sample_{idy}": timeseries_sample, "prediction": prediction_file_name}
            samples.append(sample_data)

        metadata[f"batch_{idx}"].append(samples)



    with open('predictions/{0}/{1}_{2}_batch_metadata_{3}.json'.format(prediction_directory_name, model_name, args.dataset, current_date_time), 'w') as fp:
        json.dump(metadata, fp, indent=4)

    print("samples:" + str(len(predictions * model_batch_size)))

    # for idx in range(len(test_batch_metadata)):
    #     os.mkdir("predictions/{0}/batch_{1}".format(prediction_directory_name, idx))
    # set to -1 to account for 0 mod 4 = 0 in batch_indexing
    # print(len(predictions))
    # print(f"Sample Shape: {predictions[0].shape}")
    # batch_index = 0
    # non_blank_predictions = 0

    # for idy, image in enumerate(predictions):
    #     if idy % model_batch_size == 0 and idy != 0:        
    #         batch_index += 1
        
    #     for idz in range(image.shape[-1]):
    #         images = np.array(image[:, :, idz].reshape(NETWORK_DEMS, NETWORK_DEMS, 1))
    #         img = images.reshape(NETWORK_DEMS, NETWORK_DEMS, 1).astype(dtype=np.uint8)

    #         if np.ptp(img) != 0:
    #             img_0 = array_to_img(img)
    #             filename_0 = "predictions/{0}/batch_{1}/batch_{1}_sample_{2}.tif".format(prediction_directory_name, batch_index, idy % model_batch_size)
    #             img_0.save(filename_0)
    #             non_blank_predictions+=1
    print(f"Total non-blank predictions saved: {non_blank_predictions} out of {len(predictions)} predictions")


def save_img(file_path: str, sample_file_path: str, pred, dem=NETWORK_DEMS) -> None:
    with gdal_open(sample_file_path) as f:
        mask_projection = f.GetProjection()
        mask_geo_transform = f.GetGeoTransform()

    write_mask_to_file(
        pred.reshape(dem, dem), file_path, mask_projection, mask_geo_transform
    )

if __name__ == '__main__':
    p = ArgumentParser()
    sp = p.add_subparsers()

    # Arguments for train mode
    train = sp.add_parser('train', help='Train a new model')
    train.add_argument('model', help='Name of the model to save: example_net')
    train.add_argument('dataset', nargs='?', default='dataset_calibrated')
    train.add_argument(
        '--overwrite',
        '-o',
        action='store_true',
        help='Replace the file if it exists'
    )
    train.add_argument(
        '--continue',
        '-c',
        action='store_true',
        dest='cont',
        help='Continue training from existing model'
    )
    train.add_argument('--epochs', '-e', type=int, default=10)
    train.set_defaults(func=train_wrapper)

    # Arguments for test mode
    test = sp.add_parser('test', help='Test an existing model')
    test.add_argument('model', help='Name of the trained model')
    test.add_argument('dataset', nargs='?', default='dataset_calibrated')
    test.add_argument(
        '--edit',
        '-e',
        help="Replace mask with the networks",
        action='store_true'
    )
    test.set_defaults(func=test_wrapper)

    # Parse and execute selected function
    args = p.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        p.print_help()
