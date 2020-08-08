import matplotlib.pyplot as plt
import scipy.misc
from random import randint
from PIL import Image
import colorsys
import numpy
import glob
import os
import shutil

from Parameters import data_params
from Parameters import model_params
from Parameters import hyper_params
from VesselsCoarseDataGenerator import VesselsCoarseDataGenerator
from VesselsCoarseDataReader import VesselsCoarseDataReader
from VesselsFineDataGenerator import VesselsFineDataGenerator
from VesselsFineDataReader import VesselsFineDataReader
from FileTools import empty_or_create_directory
from Unet import Unet
from FileTools import empty_or_create_directory
import imageio

#------------------------------------------------------
# generate training data
# args:
# labelled_train_db - path to train db with labels
# labelled_train_dir - path to directory with labelled 
#                      train images
# data_x_file_name - data X file name
# data_y_file_name - data Y file name
# height - height of a training image
# width - width of a training image
# vessel_radius - radius of a circle around vessel
# min_vessel_prob - minimum vessel probability
# max_vessel_prob - maximum vessel probability
# vessel_class_index - integer specifying class of 
#                      a vessel
# background_radius - radius of a circle around background
# min_background_prob - minimum background probability
# max_background_prob - maximum background probability
# background_class_index - integer specifying class of
#                          a background
# max_training_images_count - maximum training images,
#                             if -1 then take them all
# default_background_probability - default background
#                                  probability
# default_vessel_probability - default vessel
#                              probability
#------------------------------------------------------
def generate_coarse_data(   labelled_train_db,
                            labelled_train_dir,
                            data_x_file_name,
                            data_y_file_name,
                            height,
                            width,
                            vessel_radius,
                            min_vessel_prob,
                            max_vessel_prob,
                            vessel_class_index,
                            background_radius,
                            min_background_prob,
                            max_background_prob,            
                            background_class_index,
                            max_training_images_count,
                            default_background_probability,
                            default_vessel_probability):

    vessels = VesselsCoarseDataGenerator(labelled_train_db, labelled_train_dir)
    vessels.read_db()

    print("Storing X set.")
    vessels.write_training_x_data(data_x_file_name, height, width, max_training_images_count)
    print("Storing Y set.")
    vessels.write_training_y_data(data_y_file_name,
                                  height, width,
                                  vessel_radius, min_vessel_prob, max_vessel_prob, vessel_class_index,
                                  background_radius, min_background_prob, max_background_prob, background_class_index,
                                  max_training_images_count, default_background_probability, default_vessel_probability)

#------------------------------------------------------
# generate training data
# args:
# labelled_train_db - path to train db with labels
# labelled_train_dir - path to directory with labelled 
#                      train images
# data_x_file_name - data X file name
# data_y_file_name - data Y file name
# height - height of a training image
# width - width of a training image
# vessel_radius - radius of a circle around vessel
# min_vessel_prob - minimum vessel probability
# max_vessel_prob - maximum vessel probability
# vessel_class_index - integer specifying class of 
#                      a vessel
# background_radius - radius of a circle around background
# min_background_prob - minimum background probability
# max_background_prob - maximum background probability
# background_class_index - integer specifying class of
#                          a background
# max_training_images_count - maximum training images,
#                             if -1 then take them all
# default_background_probability - default background
#                                  probability
# default_vessel_probability - default vessel
#                              probability
#------------------------------------------------------
def generate_fine_data( train_dir,
                        coarse_segmentation_dir,
                        data_x_file_name,
                        data_y_file_name,
                        height,
                        width, 
                        background_class_index,
                        vessel_class_index,
                        max_training_images_count):

    vessels = VesselsFineDataGenerator(train_dir, coarse_segmentation_dir, data_params['fine_segmented_partial_file_name'])
    vessels.read()

    print("Storing X set.")
    vessels.write_training_x_data(data_x_file_name,
                                  height,
                                  width,
                                  max_training_images_count)

    print("Storing Y set.")
    vessels.write_training_y_data(data_y_file_name,
                                  height,
                                  width,                                  
                                  vessel_class_index,
                                  background_class_index,
                                  max_training_images_count)

#------------------------------------------------------
# returns full path to segmented and grey image
# segmented_dir - directory with segmentation 
#                      results
# image_file_name - original path
# step - segmentation step
#------------------------------------------------------
def get_segmented_and_grey_image_file_name(segmented_dir, image_file_name, step):

    short_image_file_name = os.path.basename(image_file_name)
    short_image_grey_file_name = short_image_file_name.replace(".jpg", "_segmented_grey.png")
    short_image_binary_file_name = short_image_file_name.replace(".jpg", "_segmented_binary.png")
    segmented_grey_file_name = "{}{}".format(segmented_dir, short_image_grey_file_name)
    segmented_binary_file_name = "{}{}".format(segmented_dir, short_image_binary_file_name)
    segmented_partially_name_prefix = "{}{}".format(segmented_dir, short_image_file_name.replace(".jpg", ""))

    return (segmented_grey_file_name, segmented_binary_file_name, segmented_partially_name_prefix)

#--------------------------------------------------------
# segment images
# unet - implementation of unet
# labelled_validate_db - path to train db with labels
# labelled_validate_dir - images to segment
# original_images_dir - input dir with original images
# segmented_dir - output dir for segmented images
# step - step size for offsetting windows
# model_path - path from which to load model weights
# batch_size - batch size
# max_labelled_images_count - maximum labelled images,
#                             if -1 then take them all
# binarisation_threshold - binarisation threshold
# padding_to_remove - how many pixels to avoid on border
# rectangle_searcher_window_size - rectangle searcher window size
#--------------------------------------------------------
def segment_images(unet,
                   labelled_validate_db,
                   labelled_validate_dir,
                   original_images_dir,
                   segmented_dir,
                   step,
                   model_path,
                   batch_size,
                   max_labelled_images_count,
                   binarisation_threshold,
                   padding_to_remove,
                   rectangle_searcher_window_size):

    unet.model.load_weights(model_path)
    vessels = VesselsCoarseDataReader(labelled_validate_db, labelled_validate_dir)
    vessels.read_db()
    empty_or_create_directory(segmented_dir)
    images_count = 0

    for image_file_name, vessels_positions_list in vessels.images_dict.items():

        (segmented_grey_file_name, segmented_binary_file_name, segmented_partially_name_prefix) = \
            get_segmented_and_grey_image_file_name(segmented_dir, image_file_name, step)

        print('input_file_name = {}, output_grey_file_name = {}, output_binary_file_name = {}'.
              format(image_file_name, segmented_grey_file_name, segmented_binary_file_name))

        unet.segment(image_file_name,
                     segmented_grey_file_name,
                     segmented_binary_file_name,
                     segmented_partially_name_prefix,
                     batch_size,
                     padding_to_remove,
                     binarisation_threshold,
                     step);

        shutil.copy2(image_file_name, segmented_dir)

        images_count += 1
        if max_labelled_images_count != -1 and images_count >= max_labelled_images_count:
            # check if we are not exceeding maximum images count
            break

#------------------------------------------------------
# save vessels color and B&W circled vessels images
# images_count - how many random images pairs
# data_x - 4D tensor with training images
# data_y - 4D tensor with B&W destination images
# element_class_index - index of a class defining element
# generated_images_dir - directory where to save images
#------------------------------------------------------
def save_images(images_count,
                data_x,
                data_y,
                height,
                width,
                element_class_index,
                generated_images_dir):

    # a sample file name just to create directory if it does not exist
    empty_or_create_directory('{}/file.jpg'.format(generated_images_dir))
   
    for i in range(0, images_count):
        index = randint(0, data_x.shape[0] - 1)
        print("Index is {}".format(index))

        # outfile_x is an original colour random file
        outfile_x_name = '{}/outfile_{}_x.jpg'.format(generated_images_dir, index)
        print("Saving image: {}".format(outfile_x_name))

        imageio.imwrite(outfile_x_name, data_x[index] * 255.0)

        # outfile_y is a B&W file with circles centred where vessels are for the random file above
        outfile_y_name = '{}/outfile_{}_y.jpg'.format(generated_images_dir, index)
        print("Saving image: {}".format(outfile_y_name))
        image_y_array = numpy.uint8(data_y[index, :, :, element_class_index] * 255.0)
        image_y = Image.fromarray(image_y_array, mode = 'L')
        # imageio.imwrite(outfile_y_name, image_y)
        scipy.misc.imsave('{}/outfile_{}_y.jpg'.format(generated_images_dir, index), image_y)

#------------------------------------------------------
# save vessels/non vessels images
# images_count - how many random images pairs
# data_x - 4D tensor with training images
# data_y - 4D tensor with B&W destination images
#------------------------------------------------------
def save_images_overlay(images_count, data_x, data_y):

    for i in range(0, images_count):
        index = randint(0, data_x.shape[0])
        print("Index is {}".format(index))

        data_x[ : , :, :, 1] = data_y[ : , : , : , 1]
        scipy.misc.imsave('outfile_x_{}.jpg'.format(index), data_x[index] * 255.0)

#------------------------------------------------------
# prints parameters used in training/segmentation
# data_params - data parmeters
# model_params - model parameters
# hyper_params - hyper parameteres
#------------------------------------------------------
def print_parameters(data_params, model_params, hyper_params):

    print("Data Parameters:")
    for key, value in data_params.items():
        print("Key: {}, Value: {}".format(key, value))

    print("Model Parameters:")
    for key, value in model_params.items():
        print("Key: {}, Value: {}".format(key, value))

    print("Hyper Parameters:")
    for key, value in hyper_params.items():
        print("Key: {}, Value: {}".format(key, value))

#------------------------------------------------------
# loads unet
# data_x_file_name - 4D tensor with training images
# data_y_file_name - 4D tensor with B&W destination 
#                    generated images
# windows_per_image_on_average - how many windows
#                                to consider
#------------------------------------------------------
def load_unet(data_x_file_name, data_y_file_name, windows_per_image_on_average):
    unet = Unet(data_x_file_name,
                data_y_file_name,
                data_params["height"],
                data_params["width"],
                hyper_params["window_size"],
                windows_per_image_on_average,
                hyper_params["percentage_train"],
                hyper_params["min_vessel_prct_window"],
                hyper_params["min_background_prct_window"],
                hyper_params["vessel_window_percentage"],
                hyper_params["background_window_percentage"],
                hyper_params["dropout"],
                hyper_params["filters_count"],
                hyper_params["kernel_size"],
                hyper_params["vessel_class_index"],
                hyper_params["background_class_index"],
                hyper_params["default_background_probability"],
                hyper_params["default_vessel_probability"],
                hyper_params["min_background_prob"],
                hyper_params["min_vessel_prob"],)

    return unet
