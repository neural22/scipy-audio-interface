__author__ = 'aloriga'

import cPickle as pickle
#import pickle

class SerializableObject:


    def save(self, path):
        """
        Save object into the file
        :param path: file name
        :return:
        """
        file_to_write = open(path, 'w')
        pickle.dump(self, file_to_write)
        file_to_write.close()


def load_obj_from_file(path):
    """
    Return an object serialized into a file
    :param path: file name
    :return: DigitalFilter
    """
    file_to_load = open(path, 'r')
    filter_obj = pickle.load(file_to_load)
    file_to_load.close()
    return filter_obj
    