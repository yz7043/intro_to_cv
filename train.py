import torch
from datasetLoader import Horse2ZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
if config.SHOW_TQDM:
    from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator

def train_horse2zebra(discriminator_H, discriminator_Z, generator_Z, generator_H, data_loader, optimizer_dis, optimizer_gen,
              L1_loss, mse_loss, dis_scaler, gen_scaler):
    loop = tqdm(data_loader, leave=True) if config.SHOW_TQDM else data_loader
    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)
        # train discriminators first
        with torch.cuda.amp.autocast():
            # train horse gen from zebra -> horse
            fake_horse = generator_H(zebra)
            D_H_real = discriminator_H(horse)
            D_H_fake = discriminator_H(fake_horse.detach())
                # real loss: we have to get closed to 1s, fake loss: we have to get closed to 0s
            D_H_real_loss = mse_loss(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse_loss(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            # train zebra gen from horse -> zebra
            fake_zebra = generator_Z(horse)
            D_Z_real = discriminator_Z(zebra)
            D_Z_fake = discriminator_Z(fake_zebra.detach())
            D_Z_real_loss = mse_loss(D_Z_real, torch.ones_like(D_H_real))
            D_Z_fake_loss = mse_loss(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # combine two discriminator loss
            D_loss = (D_H_loss + D_Z_loss) / 2

        optimizer_dis.zero_grad()
        dis_scaler.scale(D_loss).backward()
        dis_scaler.step(optimizer_dis)
        dis_scaler.update()

        # train generator
        with torch.cuda.amp.autocast():
            D_H_fake = discriminator_H(fake_horse)
            D_Z_fake = discriminator_Z(fake_zebra)
            loss_G_H = mse_loss(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse_loss(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra = generator_Z(fake_horse)
            cycle_horse = generator_H(fake_zebra)
            cycle_zebra_loss = L1_loss(zebra, cycle_zebra)
            cycle_horse_loss = L1_loss(horse, cycle_horse)

            # identity loss
            identity_zebra = generator_Z(zebra)
            identity_horse = generator_H(horse)
            identity_zebra_loss = L1_loss(zebra, identity_zebra)
            identity_horse_loss = L1_loss(horse, identity_horse)

            # add all loss
            G_loss = (loss_G_Z
                      + loss_G_H
                      + cycle_zebra_loss * config.LAMBDA_CYCLE
                      + cycle_horse_loss * config.LAMBDA_CYCLE
                      + identity_zebra_loss * config.LAMBDA_IDENTITY
                      + identity_horse_loss * config.LAMBDA_IDENTITY)
        optimizer_gen.zero_grad()
        gen_scaler.scale(G_loss).backward()
        gen_scaler.step(optimizer_gen)
        gen_scaler.update()

        if idx % 200 == 0:
            save_image(fake_horse*0.5+0.5, f"save_images/horse2zebra/fake_horse/{idx}_horse.png")
            save_image(fake_zebra*0.5+0.5, f"save_images/horse2zebra/fake_zebra/{idx}_zebra.png")

def main():
    # classify horses
    discriminator_H = Discriminator(in_ch=3).to(config.DEVICE)
    # classify zebras
    discriminator_Z = Discriminator(in_ch=3).to(config.DEVICE)
    generator_H = Generator(img_ch=3, num_residuals=9).to(config.DEVICE)
    generator_Z = Generator(img_ch=3, num_residuals=9).to(config.DEVICE)
    # initialize optimizers
    optimizer_dis = optim.Adam(
        list(discriminator_H.parameters()) + list(discriminator_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    optimizer_gen = optim.Adam(
        list(generator_Z.parameters()) + list(generator_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    L1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H, generator_H, optimizer_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Z, generator_Z, optimizer_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_H, discriminator_H, optimizer_dis, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_Z, discriminator_Z, optimizer_dis, config.LEARNING_RATE)
    # initialize the dataset
    dataset = Horse2ZebraDataset(zebra_root=config.TRAIN_ZEBRA_ROOT, horse_root=config.TRAIN_HORSE_ROOT,
                                 transform=config.transforms)
    data_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    gen_scaler = torch.cuda.amp.GradScaler()
    dis_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_horse2zebra(discriminator_H, discriminator_Z, generator_Z, generator_H, data_loader, optimizer_dis,
                          optimizer_gen, L1_loss, mse_loss, dis_scaler, gen_scaler)
        if config.SAVE_MODEL:
            save_checkpoint(generator_H, optimizer_gen, filename=config.CHECKPOINT_GEN_H%epoch)
            save_checkpoint(generator_Z, optimizer_gen, filename=config.CHECKPOINT_GEN_Z%epoch)
            save_checkpoint(discriminator_H, optimizer_dis, filename=config.CHECKPOINT_CRITIC_H%epoch)
            save_checkpoint(discriminator_Z, optimizer_dis, filename=config.CHECKPOINT_CRITIC_Z%epoch)

if __name__ == "__main__":
    main()