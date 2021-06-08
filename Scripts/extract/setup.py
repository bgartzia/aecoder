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
                 'EXTRACT_SEGMENTS':'Segmented_Defects',
                 'EXTRACT_TOT_ERROR':'Errors'}


def save_EXTRACT_LSPACES(results, res_names, out_path, model_name, bank, **kwargs):
    """
    """

    # Create dir if needed
    dir_path = os.path.join(out_path, model_name,
                            EXT_OPT_PATHS['EXTRACT_LSPACES'])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create pandas dataframe from tf.dataset
    df = pd.DataFrame(results)
    # Take only the file basename
    res_names = [os.path.basename(name) for name in res_names]

    # Bind file name columns
    res_names = pd.DataFrame(res_names)
    df = pd.concat([res_names.reset_index(drop=True), df], axis=1)

    # Write csv
    df.to_csv(os.path.join(dir_path, bank + '.csv'))


def save_EXTRACT_RAW_OUT(results, res_names, out_path, model_name, bank, **kwargs):
    """
    """

    dir_path = os.path.join(out_path, model_name,
                            EXT_OPT_PATHS['EXTRACT_RAW_OUT'], bank)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Cast images to uint8
    results = np.abs(results)
    results = results.astype(np.uint8)

    for img,name  in zip(results, res_names):
        filename = os.path.basename(name)
        cv2.imwrite(os.path.join(dir_path, filename), img)
    

def save_EXTRACT_DIFFS(results, res_names, out_path, model_name, bank, **kwargs):
    """
    """

    dir_path = os.path.join(out_path, model_name,
                            EXT_OPT_PATHS['EXTRACT_DIFFS'], bank)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Cast images to uint8
    results = np.abs(results)
    results = results.astype(np.uint8)

    for img,name  in zip(results, res_names):
        filename = os.path.basename(name)
        cv2.imwrite(os.path.join(dir_path, filename), img)


def save_EXTRACT_SEGMENTS(results, res_names, out_path, model_name, bank,
                          input_imgs=None):
    """ TODO: SEGMENTATION THRESH MUST BE SET FROM SOMEWHERE.
        IT IS NOT IMPLEMENTED YET
    """

    dir_path = os.path.join(out_path, model_name,
                            EXT_OPT_PATHS['EXTRACT_SEGMENTS'], bank)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Cast images to uint8
    results = results.astype(np.uint8)

    for img, seg, name  in zip(input_imgs, results, res_names):
        # Generate mask and detection heatmap
        _, mask = cv2.threshold(seg, 1, 255, cv2.THRESH_BINARY)
        _, inv_mask = cv2.threshold(seg, 1, 255, cv2.THRESH_BINARY_INV)
        heatmap_img = cv2.applyColorMap(seg, cv2.COLORMAP_JET)

        # If the is any pixel highlighted by the mask, ALARMA
        defect_detected = np.max(mask) > 0

        # Add heatmap to img
        img = cv2.cvtColor(img.numpy().astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img = cv2.bitwise_and(img, img, mask=inv_mask)
        heatmap_img = cv2.bitwise_and(heatmap_img, heatmap_img, mask=mask)
        heatmap_img = cv2.addWeighted(img, 1, heatmap_img, 1, 0)
        filename = os.path.basename(name)

        if not defect_detected:
            cv2.imwrite(os.path.join(dir_path, filename), heatmap_img)
        else:
            pass
            def_path = os.path.join(dir_path, 'Rejected')
            if not os.path.exists(def_path):
                os.makedirs(def_path)

            # Add red square to the image
            height, width = mask.shape[0:2]
            cv2.rectangle(heatmap_img, (0, 0), (width, height), (0, 0, 255), 3)
            # Save image
            cv2.imwrite(os.path.join(def_path, filename), heatmap_img)


def save_EXTRACT_TOT_ERROR(results, res_names, out_path, model_name, bank, **kwargs):
    """
    """

    # Create dir if needed
    dir_path = os.path.join(out_path, model_name,
                            EXT_OPT_PATHS['EXTRACT_TOT_ERROR'])

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Take only the file basename
    res_names = [os.path.basename(name) for name in res_names]

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


def code_2_exec(model, data, code):
    """ Is like a switch case but with a dictionary """
    call = {'EXTRACT_LSPACES':model.get_latent_vector,
             'EXTRACT_RAW_OUT':model.get_decoder_out,
             'EXTRACT_DIFFS':model,
             'EXTRACT_SEGMENTS':model.get_segmented_anomalies,
             'EXTRACT_TOT_ERROR':model.get_total_error
            }

    return call[code](data)



def extract(config):

    # Load model and saved weights
    model = AutoEncoder(config['data.shape'], config['model.layers'],
                        config['model.latent_dim'])
    model.load(config['model.save_path'], config['model.base_name'])
    model.seg_thresh = config['model.segment_thresh']

    # Load data 
    img_bank, img_names = load_data_for_extraction(config)

    # Call extraction
    for bank in img_bank.keys():
        for code in config['out.mode']:
            for data in img_bank[bank]:
                bank_res = code_2_exec(model=model, data=data, code=code)
                code_2_save_call[code](bank_res.numpy(), img_names[bank],
                                       config['out.path'],
                                       config['model.base_name'], bank, input_imgs=data)




