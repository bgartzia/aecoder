"""
Logic for model result extraction.
"""

import os # Path and dir handling
import cv2 # IMG saving
import numpy as np # To cast images to uint8
import pandas as pd # csv saving
from Models import AutoEncoder
from Data import load_data_for_extraction
import tensorflow as tf

# Converts config extraction CODE id to outputh directory name
EXT_OPT_PATHS = {'EXTRACT_LSPACES':'Latent_Spaces',
                 'EXTRACT_RAW_OUT':'Out_Images',
                 'EXTRACT_DIFFS':'Out_Differences',
                 'EXTRACT_SEGMENTS':'Segmented_Defects',
                 'EXTRACT_SEGMENTS_PROCESSED':'Segmented_Defects_Processed',
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
    df = pd.DataFrame(results, columns=[str(i) for i in range(1, results.shape[1]+1)])
    # Take only the file basename
    res_names = [os.path.basename(name) for name in res_names]

    # Bind file name columns
    res_names = pd.DataFrame(res_names, columns=["FileName"])
    df = pd.concat([res_names.reset_index(drop=True), df], axis=1)

    out_file = os.path.join(dir_path, bank + '.csv')
    try:
        analysed = pd.read_csv(out_file)
        analysed = pd.concat([analysed, df.reset_index(drop=True)], axis=0)
    except FileNotFoundError:
        analysed = df

    # Write csv
    analysed.to_csv(out_file, index=False)


def save_EXTRACT_RAW_OUT(results, res_names, out_path, model_name, bank, rescale, **kwargs):
    """
    """

    dir_path = os.path.join(out_path, model_name,
                            EXT_OPT_PATHS['EXTRACT_RAW_OUT'], bank)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    results = 255. * results if rescale else results
    # Cast images to uint8
    results = np.abs(results)
    results = results.astype(np.uint8)

    for img,name  in zip(results, res_names):
        filename = os.path.basename(name)
        cv2.imwrite(os.path.join(dir_path, filename), img)
    

def save_EXTRACT_DIFFS(results, res_names, out_path, model_name, bank, rescale, **kwargs):
    """
    """

    dir_path = os.path.join(out_path, model_name,
                            EXT_OPT_PATHS['EXTRACT_DIFFS'], bank)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    results = 255. * results if rescale else results
    # Cast images to uint8
    results = np.abs(results)
    results = results.astype(np.uint8)

    for img,name  in zip(results, res_names):
        filename = os.path.basename(name)
        cv2.imwrite(os.path.join(dir_path, filename), img)


def save_EXTRACT_SEGMENTS(results, res_names, out_path, model_name, bank,
                          rescale, input_imgs=None, **kwargs):

    dir_path = os.path.join(out_path, model_name,
                            EXT_OPT_PATHS['EXTRACT_SEGMENTS'], bank)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    results = 255. * results if rescale else results
    input_imgs = 255. * input_imgs if rescale else input_imgs
    # Cast images to uint8
    results = results.astype(np.uint8)

    for img, seg, name  in zip(input_imgs, results, res_names):
        # Generate mask and detection heatmap
        _, mask = cv2.threshold(seg, 1, 255, cv2.THRESH_BINARY)
        _, inv_mask = cv2.threshold(seg, 1, 255, cv2.THRESH_BINARY_INV)
        heatmap_img = cv2.applyColorMap(seg, cv2.COLORMAP_JET)

        # If the is any pixel highlighted by the mask, ALARMA
        defect_detected = np.max(mask) > 0

        # Reduce img dims by calculating mean
        img = tf.math.reduce_mean(img, axis=-1, keepdims=True)

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


def save_EXTRACT_SEGMENTS_PROCESSED(results, res_names, out_path, model_name, bank,
                                    rescale, input_imgs=None,
                                    circle_diam=6, area_thresh=10, area_max=5e3, **kwargs):

    dir_path = os.path.join(out_path, model_name,
                            EXT_OPT_PATHS['EXTRACT_SEGMENTS_PROCESSED'], bank)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    #TODO@BGARCIA: Maybe the diff images shouldnt be rescaled?
    results = 255. * results if rescale else results
    input_imgs = 255. * input_imgs if rescale else input_imgs
    # Cast images to uint8
    results = results.astype(np.uint8)

    for img, seg, name  in zip(input_imgs, results, res_names):
        # Generate mask and detection heatmap
        _, mask = cv2.threshold(seg, 1, 255, cv2.THRESH_BINARY)
        # TODO:BGARCIA DELETE LINE???
        #_, inv_mask = cv2.threshold(seg, 1, 255, cv2.THRESH_BINARY_INV)
        heatmap_img = cv2.applyColorMap(seg, cv2.COLORMAP_JET)

        # Get circle structuring element
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (circle_diam, circle_diam))

        # Apply opening on segmented mask
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel)

        # Calculate mask region stats
        n, labels, stats, cents = cv2.connectedComponentsWithStats(opened_mask,
                                                                   4, cv2.CV_32S)

        # Get labels of connected regions which area is larger than threshold
        ## Note: last column of the stats matrix is the area of the region
        # TODO@BGARCIA: Should you use vectorize instead??
        vip_labels = np.where([area_thresh < area and area < area_max\
                               for area in stats[:, -1]])[0]
        
        # Filter regions based on its area stats
        mapping_func = np.vectorize(lambda x: 1 if x in vip_labels else 0)
        mask = mapping_func(labels)
        mask = np.expand_dims(mask, axis=-1)
        inv_mask = 1 - mask

        # Is there any region left?
        defect_detected = np.max(mask) > 0

        # Reduce img dims by calculating mean (just in case you are using 
        # photometric stereo captures, or something idk)
        img = tf.math.reduce_mean(img, axis=-1, keepdims=True)
        img = img.numpy()

        # Apply masks to heatmap and image
        img = img * inv_mask
        heatmap_img = (heatmap_img * mask).astype(dtype=np.uint8)
        # Add heatmap to img
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        heatmap_img = cv2.addWeighted(img, 1, heatmap_img, 1, 0)
        filename = os.path.basename(name)

        if not defect_detected:
            cv2.imwrite(os.path.join(dir_path, filename), heatmap_img)
        else:
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

    df = pd.DataFrame(results)
    # Bind file name columns
    res_names = pd.DataFrame(res_names, columns=["FileName"])
    df = pd.concat([res_names.reset_index(drop=True), df], axis=1)

    out_file = os.path.join(dir_path, bank + '.csv')
    try:
        analysed = pd.read_csv(out_file)
        analysed = pd.concat([analysed, df], axis=0)
    except FileNotFoundError:
        analysed = df

    # Write csv
    analysed.to_csv(out_file, index=False)



code_2_save_call = \
{'EXTRACT_LSPACES':  save_EXTRACT_LSPACES,
 'EXTRACT_RAW_OUT':  save_EXTRACT_RAW_OUT,
 'EXTRACT_DIFFS':    save_EXTRACT_DIFFS,
 'EXTRACT_SEGMENTS': save_EXTRACT_SEGMENTS,
 'EXTRACT_TOT_ERROR':save_EXTRACT_TOT_ERROR,
 'EXTRACT_SEGMENTS_PROCESSED':save_EXTRACT_SEGMENTS_PROCESSED}


def code_2_exec(model, data, code):
    """ Is like a switch case but with a dictionary """
    call = {'EXTRACT_LSPACES':model.get_latent_vector,
             'EXTRACT_RAW_OUT':model.get_decoder_out,
             'EXTRACT_DIFFS':model.get_reduced_diff,
             'EXTRACT_SEGMENTS':model.get_segmented_anomalies,
             'EXTRACT_TOT_ERROR':model.get_total_error,
             'EXTRACT_SEGMENTS_PROCESSED':model.get_segmented_anomalies
            }

    return call[code](data)



def extract(config, dbg=False):

    # Load model and saved weights
    model = AutoEncoder(config['data.shape'], config['model.layers'],
                        config['model.latent_dim'])
    model.load(config['model.save_path'], config['model.base_name'])
    model.seg_thresh = config['model.segment.thresh']
    model.seg_thresh = (model.seg_thresh/255. if config['data.rescale']
                                              else model.seg_thresh)
    if dbg: print("Model loaded.\n")

    # Load data 
    if dbg: print("Loading data...")
    img_bank, img_names = load_data_for_extraction(config)
    if dbg: print("Data loaded.\n")

    # Call extraction
    if dbg: print("Extracting data...")
    for bank in img_bank.keys():
        for code in config['out.mode']:
            if dbg: print(f"Extracting {code} from {bank}...")
            for i_batch, data in enumerate(img_bank[bank]):
                if dbg: print(f"Batch n.{i_batch}...")
                bank_res = code_2_exec(model=model, data=data, code=code)
                if dbg: print(f"Saving to file...")
                until = min([(i_batch + 1) * config['data.batch_size'], len(img_names[bank])])
                code_2_save_call[code](bank_res.numpy(),
                                       img_names[bank][i_batch * config['data.batch_size']:until],
                                       config['out.path'],
                                       config['model.base_name'], bank,
                                       rescale=config['data.rescale'],
                                       input_imgs=data,
                                       circle_diam=config['model.segment.opening_r'],
                                       area_thresh=config['model.segment.area_min'],
                                       area_max=config['model.segment.area_max'])
            if dbg: print("Extracted.\n")





