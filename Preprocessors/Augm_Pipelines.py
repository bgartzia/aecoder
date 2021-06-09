""" In this file different pipelines used for data augmentation are stored
 """

import cv2
import albumentations as Ations



__v0_casq_tests_pipe = Ations.Compose([
                            Ations.CoarseDropout(max_holes=64,
                                                 max_height=2,
                                                 max_width=2,p=0.2),
                            Ations.Emboss(p=0.35),
                            Ations.GaussNoise(var_limit=(0,100),p=1),
                            Ations.HorizontalFlip(p=0.3),
                            Ations.VerticalFlip(p=0.3),
                            Ations.Rotate(limit=180,
                                          border_mode = cv2.BORDER_REPLICATE,
                                          p=1),
                            Ations.ShiftScaleRotate(scale_limit=0,
                                                    rotate_limit=0,
                                                    shift_limit=0.02,
                                                    p=1),
                            Ations.Perspective(scale=(0.001, 0.01),p=0.5),
                            Ations.RandomBrightnessContrast(brightness_limit=0.1,
                                                            contrast_limit=0.2,
                                                            p=0.2),
                            Ations.CLAHE(2, p=0.3)
                                            ])


__photom_stereo_pipe = Ations.Compose([
                            Ations.CoarseDropout(max_holes=64,
                                                 max_height=2,
                                                 max_width=2,p=0.2),
                            Ations.Emboss(p=0.35),
                            Ations.GaussNoise(var_limit=(0,100),p=1),
                            Ations.HorizontalFlip(p=0.3),
                            Ations.VerticalFlip(p=0.3),
                            Ations.Rotate(limit=180,
                                          border_mode = cv2.BORDER_REPLICATE,
                                          p=1),
                            Ations.ShiftScaleRotate(scale_limit=0,
                                                    rotate_limit=0,
                                                    shift_limit=0.02,
                                                    p=1),
                            Ations.Perspective(scale=(0.001, 0.01),
                                               p=0.5),
                            Ations.RandomBrightnessContrast(brightness_limit=0.1,
                                                            contrast_limit=0.2,
                                                            p=0.2),
                            Ations.ChannelDropout(p=0.15),
                            Ations.Solarize(threshold=100, p=0.3)
                                            ])


__all_surface_v1_pipe = Ations.Compose([
    Ations.CoarseDropout(max_holes=64,
                         max_height=2,
                         max_width=2,
                         p=0.3),
    Ations.Emboss(p=0.35),
    Ations.GaussNoise(var_limit=(0,100),
                      p=1),
    Ations.VerticalFlip(p=0.3),
    Ations.Rotate(limit=7,
                  border_mode = cv2.BORDER_REFLECT_101,
                  p=0.7),
    Ations.ShiftScaleRotate(scale_limit=0,
                            rotate_limit=0,
                            shift_limit=0.05,
                            border_mode=cv2.BORDER_REFLECT_101,
                            p=0.8),
    Ations.Perspective(scale=(0.001, 0.01),
                              p=0.5),
    Ations.RandomBrightnessContrast(brightness_limit=0.1,
                                    contrast_limit=0.2,
                                    p=0.3),
    Ations.CLAHE(2, p=0.15),
    Ations.OneOf([
        Ations.Solarize(),
        Ations.Equalize()
                 ], p=0.1),
])


Pipeline_Selector = {
            # Pipeline of the first tests made with casquillos
            'AUG_PIPE_V000':__v0_casq_tests_pipe,
            # Pipeline used for casquillos on 4channel PS images
            'AUG_PIPE_PS':__photom_stereo_pipe,
            # Pipeline used for first tests on Albedo surfaces
            'AUG_PIPE_SURF_V1':__all_surface_v1_pipe
           }
