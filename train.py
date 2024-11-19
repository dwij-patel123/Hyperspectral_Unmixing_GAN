import torch
from utils import save_some_examples
import torch.nn as nn
import torch.optim as optim
from temp_dataset import MapDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

DEVICE = torch.device("mps")


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Train Discriminator
        with torch.autocast(device_type='mps'):
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.autocast(device_type='mps'):
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * 100
            G_loss = G_fake_loss + L1
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3).to(DEVICE)
    gen = Generator(in_channels=3, features=64).to(DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    # TODO - Change L1 Loss to Wasserstein distance loss
    L1_LOSS = nn.L1Loss()

    # TODO - train directory needs to be updated
    train_dataset = MapDataset(root_dir='data/train/')
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
    )
    g_scaler = torch.GradScaler()
    d_scaler = torch.GradScaler()

    # TODO - Val directory needs to be updated
    val_dataset = MapDataset(root_dir='data/val/')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(10):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )
        # if epoch % 5 == 0:
        #     save_checkpoint(gen, opt_gen, filename='gen.pth.tar')
        #     save_checkpoint(disc, opt_disc, filename='disc.pth.tar')
        save_some_examples(gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()
