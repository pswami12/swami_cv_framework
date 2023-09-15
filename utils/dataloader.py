# from torchvision import datasets
import os
import torch
import numpy as np
import glob, random
import cv2
from albumentations import normalize

from .transform import get_transforms_CUSTOM


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, train_data_path = "train" , test_data_path = "test", extension = 'png', transform=False, calculate_mean_std=False, mean = None, std = None):
        
        self.transform = transform
        self.train=train
        self.train_data_path = os.path.join("data", train_data_path)
        self.test_data_path = os.path.join("data", test_data_path)
        self.extension = extension
        self.mean = mean
        self.std = std
        if self.train:
            image_paths, class_to_idx, classes = self.get_custom_image_paths(self.train_data_path)
            print("----------------------------")
            print(image_paths)
        else:
            image_paths, class_to_idx, classes = self.get_custom_image_paths(self.test_data_path)
        self.class_to_idx = class_to_idx
        self.classes = classes
        self.image_paths = image_paths
        if calculate_mean_std:
            self.mean, self.std = self.calculate_mean_std()
        else:
            self.mean = mean
            self.std = std
            
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image, label = self.read_image(self.image_paths[idx])
        if self.transform is not None and image is not None:
            image = normalize(image, self.mean, self.std)
            if self.train:
                image = self.transform(image=np.array(image))["image"]
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)
            
        return image, label

    def get_custom_image_paths(self, data_paths):
        # print(f"||||||||||||{os.sep}")
        image_paths = [] #to store image paths in list
        classes = [] #to store class values
        # print("//////////////////////")
        # print(glob.glob(os.path.join(data_paths, "*")))

        for data_path in glob.glob(os.path.join(data_paths, "*")):
            # print(data_path)
            classes.append(data_path.split(os.sep)[-1]) 
            # print(glob.glob(os.path.join(data_path, f"*.{self.extension}")))
            image_paths.append(glob.glob(os.path.join(data_path, f"*.{self.extension}")))
            
        image_paths = [image for images in list((image_paths)) for image in images]

        if self.train:
            random.shuffle(image_paths)

        idx_to_class = {i:j for i, j in enumerate(classes)}
        class_to_idx = {value:key for key,value in idx_to_class.items()}


        return image_paths, class_to_idx, classes

    def read_image(self, imagefile_path):
        try:
            image = cv2.imread(imagefile_path)
            # print(f"**********{imagefile_path, image}")
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = imagefile_path.split(os.sep)[-2]
            label = self.class_to_idx[label]

            return image, label
        except:
            print(imagefile_path)
            return None, None
   
    def calculate_mean_std(self):
        psum    = torch.tensor([0.0, 0.0, 0.0])
        psum_sq = torch.tensor([0.0, 0.0, 0.0])

        for i in range(0,len(self.image_paths)):
            image, _ = self.read_image(self.image_paths[i])
            if image is None:
                continue
            image = np.transpose(np.array(image), (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float)
            psum += image.sum(dim = [1, 2])
            psum_sq += (image ** 2).sum(axis = [1, 2])
        
        count = len(self.image_paths) * 224 * 224
        total_mean = psum / count
        total_var  = (psum_sq / count) - (total_mean ** 2)
        total_std  = torch.sqrt(total_var)

        return total_mean/255.0, total_std/255.0


def dataloaders(train_batch_size = None, val_batch_size = None, seed=42):

    train_transforms = get_transforms_CUSTOM(224)
    print(train_transforms)
    train_ds = CustomDataset(train=True, transform=train_transforms, mean = (0.5222, 0.4771, 0.3872), std = (0.2586, 0.2448, 0.2568))
    test_ds = CustomDataset(train=False, mean = (0.5220, 0.4850, 0.3979), std = (0.2602, 0.2461, 0.2621))

    # image, label = train_ds[1]
    # print(image)
    # cv2.imshow("Image", image)
    # cv2.waitKey()
    # image, label = test_ds[1]
    # print(image)
    # print(image)
    # cv2.imshow("Image", image)
    # cv2.waitKey()

    cuda = torch.cuda.is_available()

    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
    
    train_batch_size = train_batch_size or (128 if cuda else 64)
    val_batch_size = val_batch_size or (128 if cuda else 64)

    train_dataloader_args = dict(shuffle=True, batch_size=train_batch_size, num_workers=4, pin_memory=True)
    val_dataloader_args = dict(shuffle=True, batch_size=val_batch_size, num_workers=4, pin_memory=True) 
    print(train_dataloader_args)
    print(len(train_ds))
    train_loader = torch.utils.data.DataLoader(train_ds, **train_dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_ds, **val_dataloader_args)

    return train_loader, test_loader, train_ds.classes, train_ds.class_to_idx

# train_loader, test_laoder, classes, class_to_idx  = dataloaders(128, 128)

# print(classes)
# print(class_to_idx)
