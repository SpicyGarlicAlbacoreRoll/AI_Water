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
from PIL import Image 
# from src.plots import edit_predictions, plot_predictions

from src.model.architecture.crop_masked import create_cdl_model_masked
from keras.preprocessing.image import array_to_img

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
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

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
    predictions, test_batch_metadata = test_model_timeseries(
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
        metadata["batch_{0}".format(idx)] = []

        # for sample in batch
        for idy, sample in enumerate(batch):
            sample_timesteps = []
            
            # for timestep in sample
            for timestep in sample:
                vh_vv_pair = {"vh": timestep[0], "vv": timestep[1]}
                sample_timesteps.append(vh_vv_pair)

            # The name of the prediction produced by this sample
            prediction_file_name="prediction_batch_{0}_sample_{1}.tif".format(idx, idy)

            sample_data = {"sample{0}".format(idy): sample_timesteps, prediction: prediction_file_name}
            metadata["batch_{0}".format(idx)].append(sample_data)



    with open('predictions/{0}/{1}_{2}_batch_metadata_{3}.json'.format(prediction_directory_name, model_name, args.dataset, current_date_time), 'w') as fp:
        json.dump(metadata, fp, indent=4)

    print("samples:" + len(predictions * model_batch_size))

    # set to -1 to account for 0 mod 4 = 0 in batch_indexing
    batch_index = -1
    for idy, image in enumerate(predictions):
        if idy % model_batch_size == 0:
            batch_index += 1

        img = array_to_img(image)
        filename = "predictions/{0}/prediction_batch_{1}_sample_{2}.tif".format(prediction_directory_name, batch_index, idy)
        img.save(filename)
                
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
