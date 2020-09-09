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
from src.config import NETWORK_DEMS
from PIL import Image 
# from src.plots import edit_predictions, plot_predictions

from src.model.architecture.crop_masked import create_cdl_model_masked
from keras.preprocessing.image import array_to_img
import numpy as np

def train_wrapper(args: Namespace) -> None:
    """ Function for training a network. """
    model_name = args.model
    if args.cont:
        model = load_model(model_name)
        history = model.__asf_model_history
    else:
        model_path = path_from_model_name(model_name)
        if not args.overwrite and os.path.isfile(model_path):
            print(f"File {model_name} already exists!")
            return

        # model = create_model_masked(model_name)
        model = create_cdl_model_masked(model_name)
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], }

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

    # for batches
    for idx, batch in enumerate(test_batch_metadata):
        # print(len(batch))
        metadata["batch_{0}".format(idx)] = []
        samples = []
        # for sample in batch
        for idy, sample in enumerate(batch):
            # print(len(sample))
            timeseries_sample = {}
            # for timestep in sample
            sample_timesteps = []
            if idx ==0:
                print(sample)
            # for sub_dataset, frame_index  in sample:
                
                # for vh, vv in timeseries:
                # vh_vv_pair = {"vh": vh, "vv": vv}
            sample_timesteps = input_metadata[sample[0]][sample[1]]
            # sample_timesteps = sub_dataset
            # timeseries_mask_pair["mask"] = sample[1]
            timeseries_sample["timesteps"] = sample_timesteps
            # The name of the prediction produced by this sample

            # prediction_file_name="prediction_batch_{0}_sample_{1}.tif".format(idx, idy)
            prediction_file_name=f"predictions/{prediction_directory_name}/batch_{idx}/_sample{idy}"
            sample_data = {f"sample_{idy}": timeseries_sample, "prediction": prediction_file_name}
            samples.append(sample_data)

        metadata[f"batch_{idx}"].append(samples)



    with open('predictions/{0}/{1}_{2}_batch_metadata_{3}.json'.format(prediction_directory_name, model_name, args.dataset, current_date_time), 'w') as fp:
        json.dump(metadata, fp, indent=4)

    print("samples:" + str(len(predictions * model_batch_size)))

    for idx in range(len(test_batch_metadata)):
        os.mkdir("predictions/{0}/batch_{1}".format(prediction_directory_name, idx))
    # set to -1 to account for 0 mod 4 = 0 in batch_indexing
    
    batch_index = 0
    for idy, image in enumerate(predictions):
        if idy % model_batch_size == 0 and idy != 0:        
            batch_index += 1

        temp = np.array(image.reshape(-1, NETWORK_DEMS, NETWORK_DEMS, 1))
        for idz, frame in enumerate(range(temp.shape[0])):
            img_0 = array_to_img(temp[idz, :, :, 0].reshape(NETWORK_DEMS, NETWORK_DEMS, 1).astype(dtype=np.float32))
            # img_0 = array_to_img(temp[idz, :, :, 1].reshape(512, 512, 1).astype(dtype=np.float32))
            # img_1 = array_to_img(np.array(image[0,:,:,1].reshape(512, 512, 1)).astype(dtype=np.uint8))
            filename_0 = "predictions/{0}/batch_{1}/sample_{2}_frame_{3}_class_0.tif".format(prediction_directory_name, batch_index, idy, idz)
            # filename_1 = "predictions/{0}/batch_{1}/sample_{2}_frame_{3}_class_1.tif".format(prediction_directory_name, batch_index, idy, idz)
            img_0.save(filename_0)
            # img_1.save(filename_1)
                
    # plot_predictions(
    #     predictions, test_iter
    # )


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
