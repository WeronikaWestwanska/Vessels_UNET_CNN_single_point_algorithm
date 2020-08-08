import sqlite3
import os

class VesselsFineDataReader(object):

    def __init__(self, x_train_dir, y_train_dir, train_image_y_partial_name):

        self.x_train_dir = x_train_dir
        self.y_train_dir = y_train_dir
        self.train_image_y_partial_name = train_image_y_partial_name

    #----------------------------------------
    # reads contents of db and creates dict
    # [file_name] => (x, y)
    #---------------------------------------
    def read(self):

        self.images_dict = dict()

        for y_file_name in os.listdir(self.y_train_dir):

            if self.train_image_y_partial_name in y_file_name:

                # find corresponding Y file name
                x_file_name = y_file_name.replace(self.train_image_y_partial_name, ".jpg")
                self.images_dict[os.path.join(self.x_train_dir, x_file_name)] = os.path.join(self.y_train_dir, y_file_name)
