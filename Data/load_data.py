"""###########################################################################
    Implements the needed logic for importing the data into a tf data pipeline
###########################################################################"""

import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from random import shuffle


def read_numpy_images_from_dir(dir_path, img_format):
    """ Reads all the images with the given *img_format from *dir_path.
    """

    # Get filenames
    dir_files = glob.glob(os.path.join(dir_path, f'*.{fmt}'))
    # Reorder randomly
    shuffle(dir_files)
    # Read files as np.array
    np_imgs = np.array([cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in dir_files])

    # Add channel dim to grayscale images
    if len(np_imgs.shape) < 4:
        np_imgs = np.expand_dims(np_imgs, -1)

    return np_imgs


def setup_tf_data_pipeline(data, shuff_buff_size, batch_size):
    """ Setups the tf.data.pipeline for training/testing/whatever.
        data: Must be a np array or something like it. Could nested lists work?
        shuff_buff_size: Size of the buffer used for shuffling data before
                         creating batches.
        batch_size: data batch size.
        
        Returns:
            tf.dataset to be iterated as for batch in dataset... do things 
    """

    data_ds = tf.data.Dataset.from_tensor_slices(data)
    
    return (data_ds.shuffle(buffer_size=shuff_buff_size)
                   .batch(batch_size, drop_remainder=True)
                   .prefecth(tf.data.experimental.AUTOTUNE))


def load_data(config):
    """ Loads data as defined in the config file """


    np_train = read_numpy_images_from_dir(config['data.train'], config['data.format'])
    np_val = read_numpy_images_from_dir(config['data.test'], config['data.format'])

    return {'train': setup_tf_data_pipeline(np_train,
                                            config['train.shuffle_buffer'],
                                            config['train.batch_size']),
             'val': setup_tf_data_pipeline(np_val,
                                           np_val.shape[0],
                                           np_val.shape[0])
           }



    




