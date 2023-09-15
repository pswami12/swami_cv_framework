import albumentations as A

def get_transforms_CUSTOM(image_res):
    train_transforms = A.Compose(
        [
            A.CropAndPad(px = 16, keep_size=False),
            A.RandomCrop(width=image_res, height=image_res),
            A.CoarseDropout(2, 64, 64, 2, 64, 64,fill_value=0.4621, p=1),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        ])

    return train_transforms
