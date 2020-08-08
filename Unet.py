import scipy.misc
from random import randint
import datetime
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Conv2DTranspose, Concatenate, Permute, core
from keras.initializers import he_normal
from keras import backend as keras
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy
import math

import Parameters

class Unet:
    """Class with a UNET CNN network"""

    #---------------------------------------------------------------------------
    # ctor:
    # data_x_path - name of numpy file with X values
    # data_y_path - name of numpy file with Y values
    # height - height of an image
    # width - height of a image
    # window_size - height and width of a sliding window
    # vessel_windows_per_image_on_average - random windows per image on average
    # background_windows_per_image_on_average
    # percentage_train - how many percent of overall data is train
    # min_percentage_for_vessel_window - minimum percentage for a window to cover
    #                                    part of a vessel
    # min_percentage_for_background_window - minimum percentage for a window to cover
    #                                        part of a vessel

    # vessel_window_percentage - percentage [0:100] of vessel windows per image
    # background_window_percentage - percentage [0:100] of background windows per image
    # dropout - dropout when constructing UNET
    # filters_count - number of filter
    # kernel_size - convolution kernel size
    # vessel_class_index - index of the class: vessel
    # background_class_index - index of the class: background
    # default_background_probability - default background probability
    # default_vessel_probability - default vessel probability
    # min_background_probability - min background probability
    # min_vessel_probability - min vessel probability
    #---------------------------------------------------------------------------
    def __init__(self,
                 data_x_path,
                 data_y_path,
                 height,
                 width,
                 window_size,
                 windows_per_image_on_average,
                 percentage_train,
                 min_percentage_for_vessel_window,
                 min_percentage_for_background_window,
                 vessel_window_percentage,
                 background_window_percentage,
                 dropout,
                 filters_count,
                 kernel_size,
                 vessel_class_index,
                 background_class_index,
                 default_background_probability,
                 default_vessel_probability,
                 min_background_probability,
                 min_vessel_probability):

        self.data_x = numpy.load(data_x_path)
        self.data_y = numpy.load(data_y_path)
        self.images_count = numpy.shape(self.data_x)[0]

        self.width = width
        self.height = height
        self.window_size = window_size
        self.windows_per_image_on_average = windows_per_image_on_average
        self.percentage_train = percentage_train
        self.min_percentage_for_vessel_window = min_percentage_for_vessel_window
        self.min_percentage_for_background_window = min_percentage_for_vessel_window
        self.background_window_percentage = background_window_percentage
        self.vessel_window_percentage = vessel_window_percentage
        self.dropout = dropout
        self.vessel_class_index = vessel_class_index
        self.background_class_index = background_class_index
        self.default_background_probability = default_background_probability
        self.default_vessel_probability = default_vessel_probability
        self.min_background_probability = min_background_probability
        self.min_vessel_probability = min_vessel_probability

        self.model = self.setup_unet_model(self.window_size, filters_count, kernel_size)  

    #---------------------------------------------------------------------------
    # used for unet layer merging                                   
    #---------------------------------------------------------------------------
    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw / 2), int(cw / 2) + 1
        else:
            cw1, cw2 = int(cw / 2), int(cw / 2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch / 2), int(ch / 2) + 1
        else:
            ch1, ch2 = int(ch / 2), int(ch / 2)
    
        return (ch1, ch2), (cw1, cw2)

    #---------------------------------------------------------------------------
    # initialises Deep Neural Network U-Net architecture
    # args:
    # window_size - size of a CNN window
    # filters_count - number of filter
    # kernel_size - convolution kernel size
    # https://github.com/zhixuhao/unet/blob/master/model.py
    # https://keras.io/layers/convolutional/
    # https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
    # https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/model.py
    #----------------------------------------------------------------------------
    def setup_unet_model(self, window_size, filters_count = 64, kernel_size = 3):
        
        input_size = (self.window_size, self.window_size, 3) # RGB - that's why 3 channels
        
        # num_class = 2 means background as class = 0, foreground as class = 1, one hot encoding
        num_class = 2
        inputs = Input(input_size)
    
        concat_axis = 3
        conv1 = Conv2D(filters_count, (kernel_size, kernel_size), activation='relu', padding='same', name='conv1_1')(inputs)
        conv1 = Conv2D(filters_count, (kernel_size, kernel_size), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(filters_count * 2, (kernel_size, kernel_size), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(filters_count * 2, (kernel_size, kernel_size), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
        conv3 = Conv2D(filters_count * 4, (kernel_size, kernel_size), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(filters_count * 4, (kernel_size, kernel_size), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(filters_count * 8, (kernel_size, kernel_size), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(filters_count * 8, (kernel_size, kernel_size), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
        conv5 = Conv2D(filters_count * 16, (kernel_size, kernel_size), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(filters_count * 16, (kernel_size, kernel_size), activation='relu', padding='same')(conv5)
    
        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        ch, cw = self.get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = Conv2D(filters_count * 8, (kernel_size, kernel_size), activation='relu', padding='same')(up6)
        conv6 = Conv2D(filters_count * 8, (kernel_size, kernel_size), activation='relu', padding='same')(conv6)
    
        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        ch, cw = self.get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis) 
        conv7 = Conv2D(filters_count * 4, (kernel_size, kernel_size), activation='relu', padding='same')(up7)
        conv7 = Conv2D(filters_count * 4, (kernel_size, kernel_size), activation='relu', padding='same')(conv7)
    
        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        ch, cw = self.get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = Conv2D(filters_count * 2, (kernel_size, kernel_size), activation='relu', padding='same')(up8)
        conv8 = Conv2D(filters_count * 2, (kernel_size, kernel_size), activation='relu', padding='same')(conv8)
    
        up_conv8 = UpSampling2D(size=(2, 2))(conv8)    
        ch, cw = self.get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = Conv2D(filters_count, (kernel_size, kernel_size), activation='relu', padding='same')(up9)
        conv9 = Conv2D(filters_count, (kernel_size, kernel_size), activation='relu', padding='same')(conv9)
    
        ch, cw = self.get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        conv10 = Conv2D(num_class, (1, 1))(conv9)
    
        # this is borrowed from https://github.com/orobix/retina-unet/blob/master/src
        # good 1 but worse than good 2
        # this one is 85%
        # final11 = core.Reshape((num_class, self.windowSize * self.windowSize))(conv10)
        # final11 = core.Permute((2,1))(final11)
        # final12 = core.Activation('softmax')(final11)
        # model = Model(inputs = inputs, outputs = final12)
    
        # good 2 - almost 90%
        dense12 = core.Activation('relu')(conv10)
        dropout13 = Dropout(self.dropout)(dense12)
        final14 = core.Activation('relu')(dropout13)
        dropout15 = Dropout(self.dropout)(final14)
        final19 = core.Activation('softmax')(dropout15)
        model = Model(inputs = inputs, outputs = final19)
        
        model.summary()
        return model

    #----------------------------------------------------------------------------
    # returns a tensor of all random training, segmented windows pairs
    # windows_per_image_on_average - how many random windows to pick from each 
    #                                image on average
    # min_percentage_for_vessel_window - minimum percentage for a window to cover
    #                                    part of a vessel
    # min_percentage_for_background_window - minimum percentage for a window to cover
    #                                        part of a background
    # vessel_window_percentage - percentage [0:100] of vessel windows per image
    # we assume all data in last dimensions for x[0, 1, 2] and y[0, 1] are
    # between 0.0 and 1.0 (already normalised)
    # background_window_percentage - percentage [0:100] of background windows per image
    # we assume all data in last dimensions for x[0, 1, 2] and y[0, 1] are
    # between 0.0 and 1.0 (already normalised)
    # minimum_probability_for_background_pixel - min background probability
    # minimum_probability_for_vessel_pixel - min vessel probability
    #----------------------------------------------------------------------------
    def setup_random_windows(self,
                             windows_per_image_on_average,
                             min_percentage_for_vessel_window,
                             min_percentage_for_background_window,
                             vessel_window_percentage,
                             background_window_percentage,
                             minimum_probability_for_background_pixel,
                             minimum_probability_for_vessel_pixel):

        windows_count = self.images_count * windows_per_image_on_average
        self.data_x_windowed = numpy.zeros((windows_count, self.window_size, self.window_size, 3))
        self.data_y_windowed = numpy.zeros((windows_count, self.window_size, self.window_size, 2))

        total_vessel_windows = (windows_count * vessel_window_percentage) / 100.0
        vessel_windows_per_image = int (math.floor(total_vessel_windows / self.images_count));

        total_background_windows = (windows_count * background_window_percentage) / 100.0
        background_windows_per_image = int(math.floor(total_background_windows / self.images_count));

        # need to adjust windows_count
        windows_count = self.images_count * (vessel_windows_per_image + background_windows_per_image)

        self.data_x_windowed = numpy.zeros((windows_count, self.window_size, self.window_size, 3))
        self.data_y_windowed = numpy.zeros((windows_count, self.window_size, self.window_size, 2))

        window_index = 0
        window_pixels_count = self.window_size * self.window_size

        for image_index in range(0, self.images_count):

            # deal with the current image
            # we assume at least half windows images can be background
            # and the other half should at least contain min_percentage_for_vessel_window
            # pixels
            print("Analysing training image {} out of {} images".format(image_index, self.images_count))

            vessel_windows = 0
            background_windows = 0

            while True:

                window_found = False

                # randomize a window
                y_offset = randint(0, self.height - self.window_size)
                x_offset = randint(0, self.width - self.window_size)            

                window_x = self.data_x[ image_index, y_offset : y_offset + self.window_size,
                                        x_offset : x_offset + self.window_size, : ]  
                window_y = self.data_y[ image_index, y_offset : y_offset + self.window_size,
                                        x_offset : x_offset + self.window_size, : ]

                is_window_a_vessel = self.is_window_of_specified_class(
                    window_y, self.vessel_class_index,
                    minimum_probability_for_vessel_pixel, min_percentage_for_vessel_window, window_pixels_count)

                is_window_a_background = self.is_window_of_specified_class(
                    window_y, self.background_class_index,
                    minimum_probability_for_background_pixel, min_percentage_for_background_window, window_pixels_count)

                #print("x_offset: {}, y_offset: {}, is_window_a_vessel: {}, is_window_a_background: {}".
                #      format(x_offset, y_offset, is_window_a_vessel, is_window_a_background))

                if is_window_a_vessel and vessel_windows < vessel_windows_per_image:
                    #scipy.misc.imsave('vessel_x_{}.jpg'.format(i), window_x * 255.0)
                    #window_image_y = Image.fromarray(numpy.uint8(window_y[ :, :, 1] * 255.0), mode = 'L')
                    #scipy.misc.imsave('vessel_y_{}.jpg'.format(i), window_image_y)
                    # print("Found vessel window")
                    vessel_windows += 1
                    window_found = True
                else :
                    if is_window_a_background and background_windows < background_windows_per_image:
                        #scipy.misc.imsave('non_vessel_x_{}.jpg'.format(i), window_x * 255.0)
                        #window_image_y = Image.fromarray(numpy.uint8(window_y[ :, :, 1] * 255.0), mode = 'L')
                        #scipy.misc.imsave('non_vessel_y_{}.jpg'.format(i), window_image_y)
                        # print("Found background window")
                        background_windows += 1
                        window_found = True

                # we cannot allow to exceed the maximum number of any classes
                # windows
                if vessel_windows >= vessel_windows_per_image and background_windows >= background_windows_per_image:
                    # we do not allow further count on vessel windows now
                    break
 
                if window_found:

                    print("vw = {}, bw = {}, mvw = {}, mbw = {}, wi: {}, wc: {}, ii: {}, ic: {}".
                      format(vessel_windows,
                             background_windows,
                             vessel_windows_per_image,
                             background_windows_per_image,
                             window_index,
                             windows_count,
                             image_index,
                             self.images_count))

                    self.data_x_windowed[window_index, :, :, :] = window_x
                    self.data_y_windowed[window_index, :, :, :] = window_y
                    window_index += 1

    #----------------------------------------------------------------------------
    # train coarse network
    # batch_size - batch size
    # epochs - epochs
    # learning_rate - learning rate
    # model_path - where to save model
    #----------------------------------------------------------------------------
    def train(self, batch_size, epochs, learning_rate, model_path):

        # prepare all of training and test data by random image picking
        self.setup_random_windows(self.windows_per_image_on_average,
                                  self.min_percentage_for_vessel_window,
                                  self.min_percentage_for_background_window,
                                  self.vessel_window_percentage,
                                  self.background_window_percentage,
                                  self.min_background_probability,
                                  self.min_vessel_probability)

        # define train/test sets
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data_x_windowed, self.data_y_windowed, test_size = 1.0 - self.percentage_train / 100.0)

        self.model.compile(optimizer = Adam(lr = learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x = self.train_x,
            y = self.train_y,
            validation_data = (self.test_x, self.test_y),
            #callbacks = [EarlyStopping(monitor='val_acc', patience=5, baseline=93)],
            batch_size = batch_size,
            epochs = epochs,
            verbose = 1)
        self.model.save_weights(model_path)

    #---------------------------------------------------------------
    # checks if window contains at least min percentage of a vessel
    # args:
    # window - 3D window to check
    # class_index - class index
    # min_class_value - minimum class value
    # min_percentage_for_class_window - min percentage of pixels
    #                                 in a window > 0.0
    # window_pixels_count - number of pixels in a window
    #---------------------------------------------------------------
    def is_window_of_specified_class(self,
                                     window,
                                     class_index,
                                     min_class_value,
                                     min_percentage_for_class_window,
                                     window_pixels_count):

        pixels = (window[:, :, class_index] > min_class_value).sum()
        percentage = (pixels * 100.0) / window_pixels_count
        # print("Percentage is {}, minimum percentage is {}".format(percentage, min_percentage_for_class_window))
        if percentage >= min_percentage_for_class_window:
            return True
        else:
            return False

    #---------------------------------------------------------------
    # segments an input image using offsets
    # args:
    # input_image_name - image to segment
    # output_image_grey_name - output image in grey path as PNG
    # output_image_binary_name - output image binary path as PNG
    # segmented_partially_name_prefix - prefix for an offset based
    #                                   segmentation result
    # batch_size - batch size
    # padding - how many pixels to avoid inside the
    #           segmentation window
    # binarisation_threshold - binarisation threshold
    # step - how many pixels in each step to take
    #---------------------------------------------------------------
    def segment(self,
                input_image_name,
                output_image_grey_name,
                output_image_binary_name,
                segmented_partially_name_prefix,
                batch_size,
                padding,
                binarisation_threshold,
                step = 5):

        input_image = Image.open(input_image_name)
        input_image_array = numpy.asarray(input_image)

        # the extended image array contains window_padding pixels from each corner
        input_image_extended_array = numpy.zeros((  input_image.height + padding * 2,
                                                    input_image.width + padding * 2, 3))
        input_image_extended_array[ padding : input_image.height + padding,
                                    padding : input_image.width + padding, : ] = input_image_array / 255.0

        segmented_array_probabilities_sum = numpy.zeros((input_image.height, input_image.width))
        segmented_array_probabilities_count = numpy.zeros((input_image.height, input_image.width))
        segmented_array_probabilities_binary = numpy.zeros((input_image.height, input_image.width))

        # fill the prob_rectangles_dict
        for offset_x in range(0, self.window_size, step):
            for offset_y in range(0, self.window_size, step):

                self.segment_with_offset(batch_size,
                                         input_image.height,
                                         input_image.width,
                                         offset_x,
                                         offset_y,
                                         padding,
                                         input_image_extended_array,
                                         segmented_array_probabilities_sum,
                                         segmented_array_probabilities_count,
                                         segmented_partially_name_prefix,
                                         binarisation_threshold)

        self.store_segmented_images(segmented_array_probabilities_count,
                                    segmented_array_probabilities_sum,
                                    output_image_grey_name,
                                    output_image_binary_name,
                                    binarisation_threshold)

        

    #-------------------------------------------------------------------------------
    # stores segmented arrays as images
    # array_probabilities_count - how many pixels were processed 
    # array_probabilities_sum - sum of pixels for all segmented offsets
    # output_image_grey_name - output name for grey image
    # output_image_binary_name - output name for binary image
    # binarisation_threshold - binarisation threshold
    #-------------------------------------------------------------------------------
    def store_segmented_images(self,
                               array_probabilities_count,
                               array_probabilities_sum,
                               output_image_grey_name,
                               output_image_binary_name,
                               binarisation_threshold):

        # binarisation
        array_probabilities_binary = array_probabilities_sum / \
            array_probabilities_count

        for x in range(0, self.width):
            for y in range(0, self.height):
                if array_probabilities_binary[y][x] > binarisation_threshold:
                    array_probabilities_binary[y][x] = 255.0
                else:
                    array_probabilities_binary[y][x] = 0

        # now convert the segmented_array into shades of grey image
        # now scale the 0 to max_vessel_probability to 0 to 255
        array_probabilities_sum = array_probabilities_sum / \
            array_probabilities_count * 255.0

        # extract only foreground (vessel channel)
        output_image_grey = Image.fromarray(numpy.uint8(array_probabilities_sum), mode = 'L')
        output_image_grey.save(output_image_grey_name)
        output_image_binary = Image.fromarray(numpy.uint8(array_probabilities_binary), mode = 'L')
        output_image_binary.save(output_image_binary_name)

    #-------------------------------------------------------------------------------
    # iterates through 0 to self.width, self.height offset
    # batch_size - batch size
    # height - image height
    # width - image width
    # offset_x - offset at which window is moved to the right
    # offset_y - offset at which window is moved to the bottom
    # padding - how many pixels to avoid inside the segmentation window
    # input_image_extended_array - the extended image array contains 
    #                              window_padding pixels from each corner
    # segmented_array_probabilities_sum - 2 dimensional array [height, width]
    #                                     with sums of offseted segmentations
    # segmented_array_probabilities_count - 2 dimensional array [height, width]
    #                                       with counts of offseted segmentations
    # with a total summary of pixels' decisions for background
    # and foreground
    # segmented_partially_name_prefix - prefix for an offset based
    #                                   segmentation result
    # binarisation_threshold - binarisation threshold
    #-------------------------------------------------------------------------------
    def segment_with_offset(self,
                            batch_size,
                            height,
                            width,
                            offset_x,
                            offset_y,
                            padding,
                            input_image_extended_array,
                            segmented_array_probabilities_sum,
                            segmented_array_probabilities_count,
                            segmented_partially_name_prefix,
                            binarisation_threshold):

        # we're moving a window across image which is smaller than segmentation window
        # to avoid nasty chequered segmentation effect
        reduced_window_size = self.window_size - padding * 2

        cols_count = int(numpy.floor((width  - offset_x) / reduced_window_size))
        rows_count = int(numpy.floor((height - offset_y) / reduced_window_size))

        # windows for prediction using UNET for the current x, y offsets
        windows = numpy.zeros((rows_count * cols_count, self.window_size, self.window_size, 3))

        # temporary array (to be deleted)
        segmented_partially_array = numpy.zeros((height, width))

        local_segmented_array_probabilities_sum = numpy.zeros((height, width))
        local_segmented_array_probabilities_count = numpy.ones((height, width))

        # fill the 'windows' data
        for i in range(0, rows_count):
            for j in range(0, cols_count):

                # parts of input_image_extended_array are copied to the 'windows' tensor
                ymin = offset_y + i * reduced_window_size
                xmin = offset_x + j * reduced_window_size
                ymax = ymin + self.window_size
                xmax = xmin + self.window_size

                # need to normalize the data
                windows[i * cols_count + j, : , : , : ] = input_image_extended_array[ymin : ymax, xmin : xmax, : ]

        # now we can segment the data, by predicting the results for shifted set of windows
        predictions = self.model.predict(x = windows, batch_size = batch_size, verbose = 1, steps = None)
 
        # now expand the predictions onto the 'segmented_array' for further analysis
        for i in range(0, rows_count):
             for j in range(0, cols_count):

                ymin = offset_y + i * reduced_window_size
                xmin = offset_x + j * reduced_window_size
                ymax = ymin + reduced_window_size
                xmax = xmin + reduced_window_size

                # for each segmented_array each pixel's background and foreground
                # probabilities are appended with current window's predictions
                segmented_array_probabilities_sum[ymin : ymax, xmin : xmax] += \
                    predictions[i * cols_count + j, padding : reduced_window_size + padding,
                               padding : reduced_window_size + padding, 1]

                local_segmented_array_probabilities_sum[ymin : ymax, xmin : xmax] += \
                    predictions[i * cols_count + j, padding : reduced_window_size + padding,
                               padding : reduced_window_size + padding, 1]

                segmented_partially_array[ymin : ymax, xmin : xmax] += \
                    predictions[i * cols_count + j, padding : reduced_window_size + padding,
                                padding : reduced_window_size + padding, 1]

        segmented_array_probabilities_count[ :, : ] += 1.0

        # now try to process segmented_partially_array
        for x in range(0, self.width):
            for y in range(0, self.height):
                if segmented_partially_array[y, x] > binarisation_threshold:
                    segmented_partially_array[y, x] = 1.0
                else:
                    segmented_partially_array[y, x] = 0.0

        #local_output_image_grey_name = "{}_x{}_y{}_grey.png".format(segmented_partially_name_prefix, offset_x, offset_y)
        #local_output_image_binary_name = "{}_x{}_y{}_binary.png".format(segmented_partially_name_prefix, offset_x, offset_y)
        #self.store_segmented_images(local_segmented_array_probabilities_count,
        #                            local_segmented_array_probabilities_sum,
        #                            local_output_image_grey_name,
        #                            local_output_image_binary_name,
        #                            binarisation_threshold)

        segmented_partially_array[ :, : ] *= 255.0
        #segmented_partially_image = Image.fromarray(numpy.uint8(segmented_partially_array), mode = 'L')
        #segmented_partially_image_name = "{}_x{}_y{}.png".format(segmented_partially_name_prefix, offset_x, offset_y)
        #segmented_partially_image.save(segmented_partially_image_name)
        #print("Saving partial offset x: {}, offset y: {}, file name: {}".format(offset_x, offset_y, segmented_partially_image_name))
