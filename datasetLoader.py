import torch
from PIL import Image
import os
import config
from torch.utils.data import Dataset
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


if __name__ == "__main__":
    ds = Horse2ZebraDataset(config.TRAIN_DIR+"trainB", config.TRAIN_DIR+"trainA")
    print(ds[0][0].shape)