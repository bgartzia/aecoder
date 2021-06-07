"""
Logic for model result extraction.
"""

import os # Path and dir handling
import cv2 # IMG saving
import numpy as np # To cast images to uint8
import pandas as pd # csv saving
from Models import AutoEncoder
from Data import load_data_for_extraction

# Converts config extraction CODE id to outputh directory name
EXT_OPT_PATHS = {'EXTRACT_LSPACES':'Latent_Spaces',
                 'EXTRACT_RAW_OUT':'Out_Images',
                 'EXTRACT_DIFFS':'Out_Differences',
                 'EXTRACT_SEGMENTS':'Out_Differences',
                 'EXTRACT_TOT_ERROR':'Errors'}


def save_EXTRACT_LSPACES(results, res_names, out_path, model_name, bank):
    """
    """

    # Create dir if needed
    dir_path = os.path.join(out_path, model_name,
                            EXT_OPT_PATHS['EXTRACT_LSPACES'])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create pandas dataframe from tf.dataset
    df = pd.DataFrame(results) 
    # Bind file name columns
    res_names = pd.DataFrame(res_names)
    df = pd.concat([res_names.reset_index(drop=True), df], axis=1)

    # Write csv
    df.to_csv(os.path.join(dir_path, bank + '.csv'))


def save_EXTRACT_RAW_OUT(results, res_names, out_path, model_name, bank):
    """
    """

    dir_path = os.path.join(out_path, model_name,
                            EXT_OPT_PATHS['EXTRACT_RAW_OUT'], bank)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Cast images to uint8
    results = results.astype(np.uint8)

    for img,name  in zip(results, res_names):
        filename = os.path.basename(name)
        cv2.imwrite(os.path.join(dir_path, filename), img)
    

def save_EXTRACT_DIFFS(results, res_names, out_path, model_name, bank):
    """
    """

    dir_path = os.path.join(out_path, model_name,
                            EXT_OPT_PATHS['EXTRACT_DIFFS'], bank)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Cast images to uint8
    results = results.astype(np.uint8)

    for img,name  in zip(results, res_names):
        filename = os.path.basename(name)
        cv2.imwrite(os.path.join(dir_path, filename), img)


def save_EXTRACT_SEGMENTS(results, res_names, out_path, model_name, bank):
    """ TODO: SEGMENTATION THRESH MUST BE SET FROM SOMEWHERE.
        IT IS NOT IMPLEMENTED YET
    """

    assert False
    dir_path = os.path.join(out_path, model_name,
                            EXT_OPT_PATHS['EXTRACT_RAW_OUT'], bank)


def save_EXTRACT_TOT_ERROR(results, res_names, out_path, model_name, bank):
    """
    """

    # Create dir if needed
    dir_path = os.path.join(out_path, model_name,
                            EXT_OPT_PATHS['EXTRACT_TOT_ERROR'])

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create pandas dataframe from tf.dataset
    df = pd.DataFrame(results) 
    # Bind file name columns
    res_names = pd.DataFrame(res_names)
    df = pd.concat([res_names.reset_index(drop=True), df], axis=1)

    # Write csv
    df.to_csv(os.path.join(dir_path, bank + '.csv'))



code_2_save_call = \
{'EXTRACT_LSPACES':  save_EXTRACT_LSPACES,
 'EXTRACT_RAW_OUT':  save_EXTRACT_RAW_OUT,
 'EXTRACT_DIFFS':    save_EXTRACT_DIFFS,
 'EXTRACT_SEGMENTS': save_EXTRACT_SEGMENTS,
 'EXTRACT_TOT_ERROR':save_EXTRACT_TOT_ERROR}



def extract(config):

    # Load model and saved weights
    model = AutoEncoder(config['data.shape'], config['model.layers'],
                        config['model.latent_dim'])
    model.load(config['model.save_path'], config['model.base_name'])

    # Load data 
    img_bank, img_names = load_data_for_extraction(config)

    # Call extraction
    for bank in img_bank.keys():
        for code in config['out.mode']:
            for data in img_bank[bank]:
                bank_res = model.code_call(data=data, code=code)
                code_2_save_call[code](bank_res.numpy(), img_names[bank],
                                       config['out.path'],
                                       config['model.base_name'], bank)




