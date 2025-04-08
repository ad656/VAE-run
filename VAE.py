import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from glob import glob
import pandas as pd
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from torch.optim import lr_scheduler
import torchvision.models as models
from torch.nn.utils import spectral_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_nii_file(filepath):
    nii = nib.load(filepath)
    data = nii.get_fdata()
    return np.asarray(data, dtype=np.float32)

class PairedNIIDataset(Dataset):
    def __init__(self, anatomical_dir, fatfraction_dir, liver_mask_dir, indices=None, transform=None):
        self.anatomical_paths = sorted(glob(os.path.join(anatomical_dir, "*.nii")))
        self.fatfraction_paths = sorted(glob(os.path.join(fatfraction_dir, "*.nii")))
        self.liver_mask_paths = sorted(glob(os.path.join(liver_mask_dir, "*.nii")))
        self.transform = transform

        if len(self.anatomical_paths) == 0 or len(self.fatfraction_paths) == 0 or len(self.liver_mask_paths) == 0:
            raise ValueError("No NII files found in the specified directories.")
        if not (len(self.anatomical_paths) == len(self.fatfraction_paths) == len(self.liver_mask_paths)):
            raise ValueError("Mismatch between anatomical, fat fraction, and liver mask NII files.")

        self.slices = []
        for anatomical_path, fatfraction_path, mask_path in zip(self.anatomical_paths, self.fatfraction_paths, self.liver_mask_paths):
            volume_id = os.path.basename(anatomical_path)
            volume_id = os.path.splitext(volume_id)[0]

            anatomical_image = load_nii_file(anatomical_path)
            num_slices = anatomical_image.shape[2]
            for slice_idx in range(num_slices):
                self.slices.append((anatomical_path, fatfraction_path, mask_path, slice_idx, volume_id))

        if indices is not None:
            self.slices = [self.slices[i] for i in indices]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        anatomical_path, fatfraction_path, mask_path, slice_idx, volume_id = self.slices[idx]

        anatomical_image = load_nii_file(anatomical_path)
        fatfraction_image = load_nii_file(fatfraction_path)
        liver_mask = load_nii_file(mask_path)

        anatomical_slice = anatomical_image[:, :, slice_idx]
        fatfraction_slice = fatfraction_image[:, :, slice_idx]
        mask_slice = liver_mask[:, :, slice_idx]

        anatomical_slice = anatomical_slice * mask_slice
        fatfraction_slice = fatfraction_slice * mask_slice

        max_val_anatomical = anatomical_slice.max()
        if max_val_anatomical > 0:
            anatomical_slice = (anatomical_slice / max_val_anatomical) * 2 - 1
        else:
            anatomical_slice = anatomical_slice

        max_val_fatfraction = fatfraction_slice.max()
        if max_val_fatfraction > 0:
            fatfraction_slice = (fatfraction_slice / max_val_fatfraction) * 2 - 1
        else:
            fatfraction_slice = fatfraction_slice

        anatomical_slice = torch.from_numpy(anatomical_slice).unsqueeze(0)
        fatfraction_slice = torch.from_numpy(fatfraction_slice).unsqueeze(0)

        anatomical_slice = F.interpolate(anatomical_slice.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
        fatfraction_slice = F.interpolate(fatfraction_slice.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)

        return anatomical_slice, fatfraction_slice, volume_id

class Encoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=100):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_channels=1):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)

        self.enc1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 256, 8, 8)

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)

        dec1 = self.dec1(enc2)
        dec1 = dec1 + enc1

        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        out = self.final(dec4)

        return out

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, 16, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            spectral_norm(nn.Linear(64 * 16 * 16, 1)),
        )

    def forward(self, x):
        return self.main(x)
def display_images(anatomical, fat_fraction, reconstructed, generated, epoch, device):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    images = [anatomical, fat_fraction, generated]
    titles = ["Original Anatomical", "Original Fat Fraction", "Generated"]

    for ax, img, title in zip(axes, images, titles):
        img = img.to(device)
        img = (img + 1) / 2  # Rescale from [-1, 1] to [0, 1] for visualization
        im = ax.imshow(img.detach().squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"{title}")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    filename = f"gan_run_results_epoch_{epoch+1}.png"
    save_dir = "results"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

if __name__ == '__main__':
    num_epochs = 5000
    batch_size = 32
    latent_dim = 100

    anatomical_dir = "C:/Users/allan/Downloads/anatomical/"
    fatfraction_dir = "C:/Users/allan/Downloads/fatfrac/"
    liver_mask_dir = "C:/Users/allan/Downloads/liverseg/"

    full_dataset = PairedNIIDataset(anatomical_dir, fatfraction_dir, liver_mask_dir)
    num_slices = len(full_dataset)

    indices = list(range(num_slices))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_idx = int(num_slices * 0.8)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_dataset = PairedNIIDataset(anatomical_dir, fatfraction_dir, liver_mask_dir, indices=train_indices)
    test_dataset = PairedNIIDataset(anatomical_dir, fatfraction_dir, liver_mask_dir, indices=test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    encoder = Encoder().to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion_recon = nn.MSELoss()
    vgg = models.vgg19(pretrained=True).features[:16].to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    optimizer_E = optim.Adam(encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

    scheduler_E = lr_scheduler.CosineAnnealingLR(optimizer_E, T_max=num_epochs)
    scheduler_G = lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs)
    scheduler_D = lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=num_epochs)

    adversarial_weight = 0.5
    recon_weight = 100.0
    kl_weight = 0.001
    perceptual_weight = 1.0
    tv_weight = 0.05

    d_losses = []
    g_losses = []
    recon_losses = []
    kl_losses = []
    gp_losses = []

    n_critic = 3
    lambda_gp = 15

    for epoch in range(num_epochs):
        start_time = time.time()
        encoder.train()
        generator.train()
        discriminator.train()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_gp_loss = 0.0
        num_batches = 0

        for i, (anatomical_image, real_fatfraction_image, _) in enumerate(train_loader):
            anatomical_image = anatomical_image.to(device, non_blocking=True)
            real_fatfraction_image = real_fatfraction_image.to(device, non_blocking=True)

            for _ in range(n_critic):
                optimizer_D.zero_grad()

                mu, logvar = encoder(anatomical_image)
                z = reparameterize(mu, logvar)
                fake_fatfraction_image = generator(z).detach()

                real_validity = discriminator(real_fatfraction_image)
                fake_validity = discriminator(fake_fatfraction_image)

                gradient_penalty = compute_gradient_penalty(discriminator, real_fatfraction_image, fake_fatfraction_image)

                d_loss = torch.mean(fake_validity) - torch.mean(real_validity) + lambda_gp * gradient_penalty
                d_loss.backward()
                optimizer_D.step()

                epoch_d_loss += d_loss.item()
                epoch_gp_loss += gradient_penalty.item()
                num_batches += 1

            optimizer_E.zero_grad()
            optimizer_G.zero_grad()

            mu, logvar = encoder(anatomical_image)
            z = reparameterize(mu, logvar)
            fake_fatfraction_image = generator(z)

            fake_validity = discriminator(fake_fatfraction_image)
            g_adv_loss = -torch.mean(fake_validity) * adversarial_weight

            recon_loss = criterion_recon(fake_fatfraction_image, real_fatfraction_image) * recon_weight
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss /= anatomical_image.size(0)
            kl_loss *= kl_weight

            def perceptual_loss(gen_images, real_images):
                gen_images_rgb = gen_images.repeat(1, 3, 1, 1)
                real_images_rgb = real_images.repeat(1, 3, 1, 1)
                gen_features = vgg(gen_images_rgb)
                real_features = vgg(real_images_rgb)
                loss = F.l1_loss(gen_features, real_features)
                return loss

            p_loss = perceptual_loss(fake_fatfraction_image, real_fatfraction_image) * perceptual_weight

            def total_variation_loss(img):
                loss = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + \
                       torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
                return loss

            tv_loss = total_variation_loss(fake_fatfraction_image) * tv_weight

            loss_G = g_adv_loss + recon_loss + kl_loss + p_loss + tv_loss
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            optimizer_E.step()
            optimizer_G.step()

            epoch_g_loss += loss_G.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

        scheduler_E.step()
        scheduler_G.step()
        scheduler_D.step()

        end_time = time.time()
        epoch_duration = end_time - start_time

        avg_d_loss = epoch_d_loss / (num_batches)
        avg_g_loss = epoch_g_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        avg_gp_loss = epoch_gp_loss / (num_batches)

        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        gp_losses.append(avg_gp_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Duration: {epoch_duration:.2f}s, "
              f"D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}, "
              f"Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}, "
              f"GP: {avg_gp_loss:.4f}")
        current_lr_E = optimizer_E.param_groups[0]['lr']
        current_lr_G = optimizer_G.param_groups[0]['lr']
        current_lr_D = optimizer_D.param_groups[0]['lr']
        print(f"Learning Rates - LR_E: {current_lr_E:.6f}, LR_G: {current_lr_G:.6f}, LR_D: {current_lr_D:.6f}")

        if (epoch + 1) % 5 == 0:
            display_images(
                anatomical_image[0],
                real_fatfraction_image[0],
                fake_fatfraction_image[0],
                fake_fatfraction_image[0],
                epoch,
                device
            )

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(d_losses) + 1), d_losses, label="Discriminator Loss")
        plt.plot(range(1, len(g_losses) + 1), g_losses, label="Generator Loss")
        plt.plot(range(1, len(recon_losses) + 1), recon_losses, label="Reconstruction Loss")
        plt.plot(range(1, len(kl_losses) + 1), kl_losses, label="KL Divergence Loss")
        plt.plot(range(1, len(gp_losses) + 1), gp_losses, label="Gradient Penalty")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Losses")
        plt.savefig(f"training_losses.png")
        plt.close()

    with h5py.File('encoder_weights.h5', 'w') as f:
        for key, value in encoder.state_dict().items():
            f.create_dataset(key, data=value.cpu().numpy())

    with h5py.File('generator_weights.h5', 'w') as f:
        for key, value in generator.state_dict().items():
            f.create_dataset(key, data=value.cpu().numpy())

    with h5py.File('discriminator_weights.h5', 'w') as f:
        for key, value in discriminator.state_dict().items():
            f.create_dataset(key, data=value.cpu().numpy())

    encoder.eval()
    generator.eval()

    real_values_per_volume = {}
    pred_values_per_volume = {}

    with torch.no_grad():
        for anatomical_image, real_fatfraction_image, volume_ids in test_loader:
            anatomical_image = anatomical_image.to(device)
            real_fatfraction_image = real_fatfraction_image.to(device)

            mu, logvar = encoder(anatomical_image)
            z = reparameterize(mu, logvar)
            fake_fatfraction_image = generator(z)

            volume_id = volume_ids[0]
            real_slice = real_fatfraction_image[0].cpu().numpy().reshape(-1)
            pred_slice = fake_fatfraction_image[0].cpu().numpy().reshape(-1)

            pred_slice = (pred_slice + 1) / 2

            mask = real_slice != 0

            print(f"Volume ID: {volume_id}")
            print(f"Real slice shape: {real_slice.shape}")
            print(f"Unique values in real_slice: {np.unique(real_slice)}")
            print(f"Mask sum (number of non-zero elements): {np.sum(mask)}")

            real_slice = real_slice[mask]
            pred_slice = pred_slice[mask]

            if real_slice.size == 0 or pred_slice.size == 0:
                print(f"Empty slice after masking for Volume ID: {volume_id}")
                continue

            if volume_id not in real_values_per_volume:
                real_values_per_volume[volume_id] = []
                pred_values_per_volume[volume_id] = []

            real_values_per_volume[volume_id].extend(real_slice)
            pred_values_per_volume[volume_id].extend(pred_slice)

    results = []
    for volume_id in real_values_per_volume.keys():
        real_values = np.array(real_values_per_volume[volume_id])
        pred_values = np.array(pred_values_per_volume[volume_id])

        if len(real_values) > 0:
            real_median = np.median(real_values)
            real_iqr = np.subtract(*np.percentile(real_values, [75, 25]))
        else:
            real_median = np.nan
            real_iqr = np.nan
            print(f"No valid real values for Volume ID: {volume_id}")

        if len(pred_values) > 0:
            pred_median = np.median(pred_values)
            pred_iqr = np.subtract(*np.percentile(pred_values, [75, 25]))
        else:
            pred_median = np.nan
            pred_iqr = np.nan
            print(f"No valid predicted values for Volume ID: {volume_id}")

        results.append({
            'Volume_ID': volume_id,
            'Real_Median': real_median,
            'Real_IQR': real_iqr,
            'Pred_Median': pred_median,
            'Pred_IQR': pred_iqr
        })

    df = pd.DataFrame(results)
    df.to_excel('test_results.xlsx', index=False)
    print("Test results saved to test_results.xlsx")