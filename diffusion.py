import argparse
import logging
import sys

import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms
import matplotlib.pyplot as plt

from Diffusion.simple_unet import SimpleUNet
from Diffusion.utils import BasicDataset
from torch.utils.data import DataLoader


IMAGE_DIR = Path('./data/cars/imgs/')
SQUARE_DIM = 1280
IMG_SIZE = 64 # Shrink images to speed up training
BATCH_SIZE = 128


def load_transformed_dataset(size):
    data_transforms = [
        transforms.CenterCrop(SQUARE_DIM),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transforms = transforms.Compose(data_transforms)
    return BasicDataset(IMAGE_DIR, size=size, data_transforms=data_transforms)


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    
    # Take first image
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    
    plt.imshow(reverse_transforms(image))


def explore_diffusion(args):
    data = load_transformed_dataset(20)
    if args.show_images:
        num_samples = 20
        cols = 4
        plt.figure(figsize=(15,15))
        for i, img in enumerate(data):
            if i == num_samples:
                break
            plt.subplot(int(num_samples/cols) + 1, cols, i+1)
            show_tensor_image(img)
        plt.show()
    
    if args.noise_image:
        image = data[0]
        plt.figure(figsize=(15,15))
        plt.axis('off')
        num_images = 10
        stepsize = int(T/num_images)

        for idx in range(0, T, stepsize):
            t = torch.Tensor([idx]).type(torch.int64)
            plt.subplot(1, num_images + 1, int(idx / stepsize) + 1)
            img, noise = forward_diffusion_sample(image, t)
            show_tensor_image(img)
        plt.show()


def linear_beta_schedule(timesteps, start = 0.0001, end = 0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    """Return a specific index, t, of a list of values, vals, taking into account the batch
    dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return (
        sqrt_alphas_cumprod_t.to(device) * x_0 + 
        sqrt_one_minus_alphas_cumprod_t.to(device) * noise,
        noise.to(device)
    )


# Define beta schedule
T = 200
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for the closed form solution
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, t.device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


@torch.no_grad()
def sample_timestep(model, x, t):
    """Calls the model to predict the noise in the image and returns the denoised image.
    Applies noise to this image, if we are not in the last step
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, device):
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img - sample_timestep(model, img, t)
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)
            show_tensor_image(img.detach().cpu())
    plt.show()


def train_diffusion(device, args):
    logging.info("Training simple diffusion UNet")
    model = SimpleUNet()
    model.to(device)
    logging.info(f"Num params: {sum(p.numel() for p in model.parameters())}")

    # Hyperparameters
    optimizer = Adam(model.parameters(), lr=1e-3)
    epochs = 100

    # Data
    data = load_transformed_dataset(128)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model, batch, t)
            loss.backward()
            optimizer.step()

            if epoch % 1 == 0 and step == 0:
                logging.info(f"Epoch: {epoch}, step: {step}, loss: {loss.item():.4f}")
                sample_plot_image(model, device)

def main(args):
    parser = argparse.ArgumentParser(description="Diffusion model")
    commands = parser.add_subparsers(dest="cmd")
    
    diffusion_cmd = commands.add_parser("Explore", help="Explore diffusion data")
    diffusion_cmd.add_argument("--show_images", "-i", action="store_true", help="Show Images")
    diffusion_cmd.add_argument("--noise_image", "-n", action="store_true", help="Simulate forward diffusion")
    diffusion_cmd.set_defaults(action=explore_diffusion)

    train_cmd = commands.add_parser("train", help="Train a ddpm diffusion network")
    train_cmd.set_defaults(action=train_diffusion)
    
    args = parser.parse_args()
    if not hasattr(args, "action"):
        parser.print_help()
        return 1

    args.action(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))