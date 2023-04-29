import torch
from PIL import Image
import os
import config
from torch.utils.data import Dataset
import cv2
import numpy as np


class Horse2ZebraDataset(Dataset):
    def __init__(self, zebra_root, horse_root, transform=None):
        self.zebra_root = zebra_root
        self.horse_root = horse_root
        self.transform = transform

        self.zebra_imgs = os.listdir(zebra_root)
        self.horse_imgs = os.listdir(horse_root)

        self.zebra_len = len(self.zebra_imgs)
        self.horse_len = len(self.horse_imgs)

        self.dataset_len = max(self.zebra_len, self.horse_len)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # use module operator to avoid overflow
        zebra_img = self.zebra_imgs[idx % self.zebra_len]
        horse_img = self.horse_imgs[idx % self.horse_len]
        # get full directory to the image
        zebra_dir = os.path.join(self.zebra_root, zebra_img)
        horse_dir = os.path.join(self.horse_root, horse_img)
        # convert to np tensor
        zebra_img = np.array(Image.open(zebra_dir).convert("RGB"))
        horse_img = np.array(Image.open(horse_dir).convert("RGB"))

        if self.transform:
            aug = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = aug["image"]
            horse_img = aug["image0"]
        return zebra_img, horse_img

class Edge2ColoredDataset(Dataset):
    def __init__(self, A_root, B_root, transform=None):
        self.A_root = A_root
        self.B_root = B_root
        self.A_imgs = os.listdir(A_root)
        self.B_imgs = os.listdir(B_root)
        self.A_len = len(self.A_imgs)
        self.B_len = len(self.B_imgs)

        self.dataset_len = max(self.A_len, self.B_len)


    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        A_img = self.A_imgs[idx]
        B_img = self.B_imgs[idx]
        A_dir = os.path.join(self.A_root, A_img)
        B_dir = os.path.join(self.B_root, B_img)
        A_img = self._convertToEdge(A_dir)
        B_img = np.array(Image.open(B_dir). convert("RGB"))
        return A_img, B_img

    def _convertToEdge(self, img_dir):
        img = cv2.imread(img_dir)
        edge = cv2.Canny(img, 150, 200)
        edge_img = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
        return np.array(edge_img)

if __name__ == "__main__":
    ds = Horse2ZebraDataset(config.TRAIN_ZEBRA_ROOT, config.TRAIN_HORSE_ROOT)
    print(ds[0][0].shape)
    # ds = Edge2ColoredDataset(config.EDGE_RGB_ROOT_A, config.EDGE_RGB_ROOT_B)
    # print(ds[0][0].shape)
    # print(ds[0][1].shape)
