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
from VesselsCoarseDataReader import VesselsCoarseDataReader

class VesselsCoarseDataGenerator(VesselsCoarseDataReader):

    #---------------------------------------------------------
    # writes training x data (color images as numpy)
    # args:
    # data_x_file_name - location where to data X
    # height - height of a typical image with vessels
    # width - width of a typical image with vessels
    # max_training_images_count - maximum training images, 
    #                    if -1 then take them all
    #---------------------------------------------------------
    def write_training_x_data(self, data_x_file_name, height, width, max_training_images_count):

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
        for image_name, vessels_positions_dict in self.images_dict.items():

            image = PIL.Image.open(image_name)
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
    # default_background_probability - default background
    #                                  probability
    # default_vessel_probability - default vessel probability
    #-------------------------------------------------------
    def write_training_y_data(self,
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

        radial_vessel = RadialElement(min_vessel_prob, max_vessel_prob, vessel_radius)
        radial_background = RadialElement(min_background_prob, max_background_prob, background_radius)

        image_index = 0

        # Any non background and non vessel explicitly declared pixel will have probability 0.5.
        y_data[:, :, :, vessel_class_index] = default_vessel_probability
        y_data[:, :, :, background_class_index] = default_background_probability

        for image_name, elements_positions_list in self.images_dict.items():

            # list of vessels pixels positions
            vessel_positions_list = list()

            # list of background pixels positions.
            background_positions_list = list()

            for x, y, element in elements_positions_list:

                # distinguish between element class: 0 - element, 1 - background?
                if element == vessel_class_index:
                    vessel_positions_list.append((x,y))

                elif element == background_class_index:
                    background_positions_list.append((x,y))

            current_vessels_heatmap = ElementHeatMap(radial_vessel, vessel_positions_list, height, width)
            current_vessels_heatmap_array = current_vessels_heatmap.get_heatmap()

            current_background_heatmap = ElementHeatMap(radial_background, background_positions_list, height, width)
            current_background_heatmap_array = current_background_heatmap.get_heatmap()

            # Copy contents of the current_vessels_heatmap_array to training_y_values.

            # First - process background circles.
            for x in range(0, width):
                for y in range(0, height):

                    background_prob = current_background_heatmap_array[y][x]
                    if (background_prob > 0.0):
                        #print("x: {}, y: {}, background prob value: {}".format(x, y, background_prob))
                        y_data[image_index, y, x, vessel_class_index] = 1 - background_prob
                        y_data[image_index, y, x, background_class_index] = background_prob

            # Second - process vessels circles.
            for x in range(0, width):
                for y in range(0, height):

                    vessel_prob = current_vessels_heatmap_array[y][x]
                    if (vessel_prob > 0.0):
                        #print("x: {}, y: {}, vessel prob value: {}".format(x, y, vessel_prob))
                        y_data[image_index, y, x, vessel_class_index] = vessel_prob
                        y_data[image_index, y, x, background_class_index] = 1 - vessel_prob

            vessels_pixels_count = numpy.sum(y_data[image_index, :, :, vessel_class_index] > 0.5)
            background_pixels_count = numpy.sum(y_data[image_index, :, :, background_class_index] > 0.5)

            #print("Processed images count = {}, vessels pixels count: {}, background pixels count: {}".
            #      format(processed_images_count, vessels_pixels_count, background_pixels_count))
            image_index += 1

            # check if we are not exceeding maximum images count
            if image_index >= images_total_count:
                break
                    
        numpy.save(data_y_file_name, y_data)
