import albumentations as Ations
import os
import cv2
from .Augm_Pipelines import Pipeline_Selector


class Augmentor:
    
    def __init__(self, out_path, out_shape, pipe_code):
        self.out_path = out_path
        self.out_shape = out_shape
        self.augm_pipeline = Pipeline_Selector[pipe_code]


    def augmentate(self, times, path, splitted, format='png'):
        """ Times: indicates number of images that are going to be created from an
                   OG image.
            path: path to the place the OG images are placed.
            train_test: dict with train test keys and filenames as values.
            out: path to the folder in which the resultant train and test folders
                 are placed.
        """

        for split in splitted.keys():
            out_dir = os.path.join(self.out_path, split)
            for filename in splitted[split]:
                filepath = os.path.join(path, f'{filename}.{format}')
                og = cv2.imread(filepath,
                                cv2.IMREAD_UNCHANGED)
                if og is None:
                    raise FileNotFoundError(f'\nDidn\'t found {filepath}\nCrashing...')
                rs_og = cv2.resize(og, self.out_shape[0:2],
                                   interpolation=cv2.INTER_LINEAR)

                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                cv2.imwrite(os.path.join(out_dir, f'OG_{filename}.{format}'), rs_og)

                if split == 'train':
                    for i_aug in range(times):
                        pipeline_out = self.augm_pipeline(image=rs_og)
                        augmented = pipeline_out['image']
                        out_filename = os.path.join(out_dir,
                                            f'augm_{i_aug}_{filename}.{format}')
                        cv2.imwrite(out_filename, augmented)


