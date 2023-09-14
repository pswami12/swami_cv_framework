import albumentations as A
import glob
import random

def get_transforms_CUSTOM(train_mean_std, test_mean_std, image_res):
    train_transforms = A.Compose(
        [
            A.CropAndPad(px = 16, keep_size=False),
            A.RandomCrop(width=image_res, height=image_res),
            A.CoarseDropout(2, 64, 64, 2, 64, 64,fill_value=0.4621, p=1),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RGBShift(r_shift_limit = 15, g_shift_limit = 15, b_shift_limit = 15, p = 0.5),
            A.RandomBrightnessContrast(p = 0.5),
            A.Normalize(train_mean_std[0], train_mean_std[1]),
            A.HueSaturationValue(hue_shift_limit = 0.2, sat_shift_limit = 0.2, val_shift_limit = 0.2, p = 0.5)
        ])



    val_transforms = A.Compose(
        [
            A.Normalize(mean = test_mean_std[0], std = test_mean_std[1]),
        ])
    return train_transforms, val_transforms
