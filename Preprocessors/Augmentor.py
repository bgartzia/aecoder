# TODO@BGARCIA
import albumentations as Ations
import os
import numpy as np
import pandas as pd
from cv2 import (imread, resize, IMREAD_UNCHANGED, INTER_LINEAR, imwrite)


class Augmentor:
    
    def __init__(self, out_path, out_shape):
        self.out_path = out_path
        self.out_shape = out_shape
        self.augm_pipeline =  Ations.Compose([
                                    Ations.CoarseDropout(max_holes=64,
                                                         max_height=2,
                                                         max_width=2,p=0.2),
                                    Ations.Emboss(p=0.35),
                                    Ations.GaussNoise(var_limit=(0,100),p=1),
                                    Ations.HorizontalFlip(p=0.3),
                                    Ations.VerticalFlip(p=0.3),
                                    Ations.Rotate(limit=180,
                                                  border_mode = cv2.BORDER_REPLICATE,p=1),
                                    Ations.ShiftScaleRotate(scale_limit=0,
                                                            rotate_limit=0,
                                                            shift_limit=0.01,
                                                            p=1),
                                    Ations.Perspective(scale=(0.001, 0.01),p=0.5),
                                    Ations.RandomBrightnessContrast(brightness_limit=0.1,
                                                                    contrast_limit=0.2,
                                                                    p=0.1),
                                    Ations.CLAHE(2, p=0.3)
                                             ])



    def augmentate(times, path, splitted):
        """ Times: indicates number of images that are going to be created from an
                   OG image.
            path: path to the place the OG images are placed.
            train_test: dict with train test keys and filenames as values.
            out: path to the folder in which the resultant train and test folders
                 are placed.
        """

        for split in splitted.keys():
            for filename in splitted[split]:
                og = imread(os.path.join(path, filename), IMREAD_UNCHANGED) 
                rs_og = resize(og, self.out_shape, interpolation=INTER_LINEAR)
                imwrite(os.path.join(self.out_path, split, 'OG_' + filename), rs_og)

                if split == 'train'
                    for i_aug in range(times):
                        pipeline_out = self.augm_pipeline(image=rs_og)
                        augmented = pipeline_out['image']
                        out_filename = os.path.join(self.out_path, split,
                                                    f'augm_{i_aug}' + filename) 
                        imwrite(out_filename, augmented)


