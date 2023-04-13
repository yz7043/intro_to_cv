import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "horse2zebra/"
VAL_DIR = "horse2zebra/"
TRAIN_HORSE_ROOT = "horse2zebra/trainA/"
TRAIN_ZEBRA_ROOT = "horse2zebra/trainB/"

BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
SHOW_TQDM = True
CHECKPOINT_GEN_H = "checkpoints/horse2zebra/%d_genh.pth.tar"
CHECKPOINT_GEN_Z = "checkpoints/horse2zebra/%d_genz.pth.tar"
CHECKPOINT_CRITIC_H = "checkpoints/horse2zebra/%d_critich.pth.tar"
CHECKPOINT_CRITIC_Z = "checkpoints/horse2zebra/%d_criticz.pth.tar"

FAKE_HORSE_PATH = "save_images/horse2zebra/fake_horse/%d_%d_horse.png"
FAKE_ZEBRA_PATH = "save_images/horse2zebra/fake_zebra/%d_%d_zebra.png"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
