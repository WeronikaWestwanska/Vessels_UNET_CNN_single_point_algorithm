import sqlite3
import os

class VesselsCoarseDataReader(object):
    """Class with a collection of mappings image -> [(x1,y1), (x2, y2), ..., (xn, yn)]"""
    """and methods to store heatmapped vessels onto a 2D array"""
    # https://fairyonice.github.io/Learn-about-ImageDataGenerator.html
    def __init__(self, db_name, labelled_dir):
        self.db_name = db_name
        self.labelled_dir = labelled_dir

    #----------------------------------------
    # reads contents of db and creates dict
    # [file_name] => (x, y)
    #---------------------------------------
    def read_db(self):
        connection = sqlite3.connect(self.db_name)
        cursor = connection.cursor()
        self.images_dict = dict()
        images_numbers_dict = dict()
        
        for row in cursor.execute('select * from imgs'):
            index, file_name = row
            images_numbers_dict[index] = file_name

        for row in cursor.execute('select * from labels'):
            image_num, x, y, object = row
            # print("image_num: {}, x = {}, y = {}".format(image_num, x, y))
            path = os.path.join(self.labelled_dir, images_numbers_dict[image_num])
            if path not in self.images_dict.keys():
                self.images_dict[path] = list()
            
            self.images_dict[path].append((x, y, object))
