import tensorflow
import sqlite3
import numpy
import PIL
import os
from keras.preprocessing.image import ImageDataGenerator
from random import randint
import keras
import glob

from FileTools import empty_or_create_directory

from ElementHeatMap import RadialElement
from ElementHeatMap import ElementHeatMap
from VesselsFineDataReader import VesselsFineDataReader

class VesselsFineDataGenerator(VesselsFineDataReader):

    #---------------------------------------------------------
    # writes training x data (color images as numpy)
    # args:
    # data_x_file_name - location where to data X
    # height - height of a typical image with vessels
    # width - width of a typical image with vessels
    # max_training_images_count - maximum training images, 
    #                    if -1 then take them all
    #---------------------------------------------------------
    def write_training_x_data(self,
                              data_x_file_name,
                              height,
                              width,
                              max_training_images_count):

        # setup numpy array to store vessels heatmaps
        images_total_count = len(self.images_dict.keys())

         # check if we are not exceeding maximum images count
        if max_training_images_count != -1 and max_training_images_count < images_total_count:
            images_total_count = max_training_images_count

        x_data = numpy.zeros(shape = (images_total_count, height, width, 3))

        # setup empty directory
        empty_or_create_directory(data_x_file_name)

        # go through collection of images and 
        # store them via augmentation
        processed_images_count = 0
        for x_image_name, y_image_name in self.images_dict.items():

            image = PIL.Image.open(x_image_name)
            # normalising the data for 3 channels
            image_as_array = numpy.asarray(image) / 255.0
            numpy.copyto(x_data[processed_images_count], image_as_array)     

            processed_images_count += 1

            # check if we are not exceeding maximum images count
            if processed_images_count >= images_total_count:
                break
                    
        numpy.save(data_x_file_name, x_data)

    #-------------------------------------------------------
    # writes training y data (vessels heatmaps)
    # images to directory with
    # data_y_file_name - location where to store heat map
    # height - height of a typical image with vessels
    # width - width of a typical image with vessels
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
    # default_vessel_probability - default vessel probability
    #-------------------------------------------------------
    def write_training_y_data(self,
                              data_y_file_name,
                              height,
                              width,
                              vessel_class_index,                             
                              background_class_index,
                              max_training_images_count):

        # setup numpy array to store vessels heatmaps
        images_total_count = len(self.images_dict.keys())

        # check if we are not exceeding maximum images count
        if max_training_images_count != -1 and max_training_images_count < images_total_count:
            images_total_count = max_training_images_count

        y_data = numpy.zeros(shape = (images_total_count, height, width, 2))

        # setup empty directory
        empty_or_create_directory(data_y_file_name)

        # go through collection of images and 
        # store them via augmentation
        processed_images_count = 0
        for x_image_name, y_image_name in self.images_dict.items():

            image = PIL.Image.open(y_image_name)
            # normalising the data for 1 channel
            image_as_array = numpy.asarray(image) / 255.0

            for x in range(0, image.width):
                for y in range(0, image.height):

                    if image_as_array[y][x] == 0: 

                        y_data[processed_images_count][y][x][vessel_class_index] = 0
                        y_data[processed_images_count][y][x][background_class_index] = 1

                    else:

                        y_data[processed_images_count][y][x][background_class_index] = 0
                        y_data[processed_images_count][y][x][vessel_class_index] = 1

            processed_images_count += 1
                    
        numpy.save(data_y_file_name, y_data)
