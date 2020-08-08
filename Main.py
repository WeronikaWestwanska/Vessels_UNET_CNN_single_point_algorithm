from random import randint
import numpy

from Parameters import data_params
from Parameters import model_params
from Parameters import hyper_params
from VesselsCoarseDataGenerator import VesselsCoarseDataGenerator
from VesselsFineDataGenerator import VesselsFineDataGenerator
from Unet import Unet

from Tools import generate_coarse_data
from Tools import generate_fine_data
from Tools import save_images
from Tools import segment_images
from Tools import print_parameters
from Tools import load_unet
from FileTools import empty_or_create_directory

import argparse

#----------------------------
# main routine
#----------------------------

parser = argparse.ArgumentParser(description='Vessels Segmentation.')

parser.add_argument('-g1', '--generate-coarse',
                    help='with this option program will only generate coarse data for the coarse training',
                    dest='generate_coarse_data', action='store_true')
parser.add_argument('-t1', '--train-coarse',
                    help='with this option the program will do coarse trainining of the Unet model and store it',
                    dest='train_coarse', action='store_true')
parser.add_argument('-s1', '--segment-coarse',
                    help='with this option the program will do coarse segmentation of images',
                    dest='segment_coarse', action='store_true')

parser.add_argument('-g2', '--generate-fine',
                    help='with this option program will only generate fine data for the fine training',
                    dest='generate_fine_data', action='store_true')
parser.add_argument('-t2', '--train-accurate',
                    help='with this option the program will do fine trainining of the Unet model and store it',
                    dest='train_fine', action='store_true')
parser.add_argument('-s2', '--segment-accurate',
                    help='with this option the program will do fine segmentation of images',
                    dest='segment_fine', action='store_true')

args = parser.parse_args()

# print current parameters
print_parameters(data_params, model_params, hyper_params)

# paths to files with numpy data 
data_x_file_name = 'data/train_common/vessels_x.npy'

data_coarse_y_file_name = 'data/train_coarse_vr{}_br{}/vessels_coarse_y_vr{}_br{}.npy'.format(
    hyper_params["vessel_radius"], hyper_params["background_radius"], hyper_params["vessel_radius"], hyper_params["background_radius"])

data_fine_y_file_name = 'data/train_fine/vessels_fine.npy'

generated_images_dir = 'data/train_coarse_vr{}_br{}/sample_images'.format(
    hyper_params["vessel_radius"], hyper_params["background_radius"])

# path to trained model
coarse_model_path = 'data/train_coarse_vr{}_br{}/coarse_model_weights_vr{}_br{}_w{}.h5'.format(
    hyper_params["vessel_radius"], hyper_params["background_radius"], hyper_params["vessel_radius"],
    hyper_params["background_radius"], hyper_params["window_size"])
fine_model_path = 'data/train_fine/train_fine.h5'


if args.generate_coarse_data:

    # step 1 - generate coarse
    generate_coarse_data(   data_params["labelled_train_db"],
                            data_params["labelled_train_dir"],
                            data_x_file_name,
                            data_coarse_y_file_name,
                            height = data_params["height"],
                            width = data_params["width"],
                            vessel_radius = hyper_params["vessel_radius"],
                            min_vessel_prob = hyper_params["min_vessel_prob"],
                            max_vessel_prob = hyper_params["max_vessel_prob"],
                            vessel_class_index = hyper_params["vessel_class_index"],
                            background_radius = hyper_params["background_radius"],
                            min_background_prob = hyper_params["min_background_prob"],
                            max_background_prob = hyper_params["max_background_prob"],
                            background_class_index = hyper_params["background_class_index"],
                            max_training_images_count = hyper_params["max_training_images_count"],
                            default_background_probability = hyper_params["default_background_probability"],
                            default_vessel_probability = hyper_params["default_vessel_probability"])
   
    # verify data visually
    data_x = numpy.load(data_x_file_name)
    data_y = numpy.load(data_coarse_y_file_name)

    save_images(50,
                data_x,
                data_y,
                data_params["height"],
                data_params["width"],
                hyper_params["vessel_class_index"],
                generated_images_dir)

if args.train_coarse:

    # step 2 - train    
    unet = load_unet(data_x_file_name,
                     data_coarse_y_file_name,
                     hyper_params["windows_per_image_on_average_coarse"])

    # train is only enabled via command line
    unet.train( model_params["batch_size"],
                model_params["epochs"],
                hyper_params["learning_rate"],
                coarse_model_path)

if args.segment_coarse:

    unet = load_unet(data_x_file_name,
                     data_coarse_y_file_name,
                     hyper_params["windows_per_image_on_average_coarse"])

    # step 3 - segment images
    segment_images( unet,
                    data_params["labelled_validate_db"],
                    data_params["labelled_validate_dir"],
                    data_params["labelled_train_dir"],
                    data_params["segmented_coarse_dir"],
                    hyper_params["sliding_window_step"],
                    coarse_model_path,
                    model_params["batch_size"],
                    hyper_params["max_testing_images_count"],
                    hyper_params["binarisation_threshold"],
                    hyper_params["padding_to_remove"],
                    hyper_params["rectangle_searcher_window_size"])

if args.generate_fine_data:

    # step 4 - generate coarse
    generate_fine_data( data_params["labelled_train_dir"],
                        data_params["segmented_coarse_dir"],
                        data_x_file_name,
                        data_fine_y_file_name,
                        height = data_params["height"],
                        width = data_params["width"],
                        background_class_index = hyper_params["background_class_index"],
                        vessel_class_index = hyper_params["vessel_class_index"],
                        max_training_images_count = hyper_params["max_training_images_count"])

if args.train_fine:

    # step 5 - train fine
    unet = load_unet(data_x_file_name,
                     data_fine_y_file_name,
                     hyper_params["windows_per_image_on_average_fine"])

    # train is only enabled via command line
    unet.train( model_params["batch_size"],
                30,
                hyper_params["learning_rate"],
                fine_model_path)

if args.segment_fine:

    unet = load_unet(data_x_file_name,
                     data_fine_y_file_name,
                     hyper_params["windows_per_image_on_average_fine"])

    # step 6 - fine segment images
    segment_images( unet,
                    data_params["labelled_validate_db"],
                    data_params["labelled_validate_dir"],
                    data_params["labelled_train_dir"],
                    data_params["segmented_fine_dir"],
                    hyper_params["sliding_window_step"],
                    fine_model_path,
                    model_params["batch_size"],
                    hyper_params["max_testing_images_count"],
                    hyper_params["binarisation_threshold"],
                    hyper_params["padding_to_remove"],
                    hyper_params["rectangle_searcher_window_size"])


print("Finished")
