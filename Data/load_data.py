"""###########################################################################
    Implements the needed logic for importing the data into a tf data pipeline
###########################################################################"""

import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from random import shuffle


def read_numpy_images_from_dir(dir_path, img_format, shuff=True, names=False,
                               rescale=True):
    """ Reads all the images with the given *img_format from *dir_path.
    """

    # Get filenames
    dir_files = glob.glob(os.path.join(dir_path, f'*.{img_format}'))
    # Reorder randomly
    if shuff: shuffle(dir_files)
    # Read files as np.array
    np_imgs = np.array([cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in dir_files])

    # Add channel dim to grayscale images
    if len(np_imgs.shape) < 4:
        np_imgs = np.expand_dims(np_imgs, -1)

    # Cast to float32
    np_imgs = np_imgs.astype(dtype=np.float32)

    # Rescale images (if wanted)
    if rescale: np_imgs = np_imgs/255.

    if names:
        return np_imgs.astype(dtype=np.float32), dir_files
    else:
        return np_imgs.astype(dtype=np.float32)


def setup_tf_data_pipeline(data, shuff_buff_size, batch_size, shuffle=True):
    """ Setups the tf.data.pipeline for training/testing/whatever.
        data: Must be a np array or something like it. Could nested lists work?
        shuff_buff_size: Size of the buffer used for shuffling data before
                         creating batches.
        batch_size: data batch size.
        
        Returns:
            tf.dataset to be iterated as for batch in dataset... do things 
    """

    data_ds = tf.data.Dataset.from_tensor_slices(data)

    if shuffle:
        data_ds = data_ds.shuffle(buffer_size=shuff_buff_size)

    return (data_ds.batch(batch_size, drop_remainder=True)
                   .prefetch(tf.data.experimental.AUTOTUNE))


def load_data(config):
    """ Loads data as defined in the config file """


    np_train = read_numpy_images_from_dir(config['data.train'], config['data.format'])
    np_val = read_numpy_images_from_dir(config['data.test'], config['data.format'])

    return {'train': setup_tf_data_pipeline(np_train,
                                            config['train.shuffle_buffer'],
                                            config['train.batch_size']),
             'val': setup_tf_data_pipeline(np_val,
                                           np_val.shape[0],
                                           config['train.batch_size'])
           }


def load_data_for_extraction(config):
    """ Loads data as in load_data but without shuffling. Returns a list
        instead of a dict """

    img_bank = {}
    name_bank = {}
    for path in config['data.in']:
        np_imgs, nm_imgs = read_numpy_images_from_dir(path, config['data.format'],
                                                      shuff=False, names=True,
                                                      rescale=config['data.rescale'])
        key = path.split('/')[-1]
        img_bank[key] = setup_tf_data_pipeline(np_imgs, 0, np_imgs.shape[0],
                                               shuffle=False)
        name_bank[key] = nm_imgs

    return img_bank, name_bank

    

    




