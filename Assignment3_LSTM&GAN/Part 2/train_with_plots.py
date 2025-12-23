import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

from my_gan import Generator, Discriminator


def train_with_logging(dataloader, discriminator, generator, optimizer_G, optimizer_D, device, args):
    """Train GAN and log losses for plotting."""
    criterion = nn.BCELoss()
    
    # Lists to store losses
    d_losses = []
    g_losses = []
    iterations = []
    d_real_acc = []
    d_fake_acc = []
    d_avg_acc = []
    g_acc = []
    
    batches_per_epoch = len(dataloader)
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            
            # Ground truths
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Move real images to device
            real_imgs = imgs.to(device)

            # -----------------
            # Train Discriminator
            # -----------------
            optimizer_D.zero_grad()
            
            # Loss for real images
            real_output = discriminator(real_imgs)
            d_loss_real = criterion(real_output, real_labels)
            
            # Generate fake images
            z = torch.randn(batch_size, args.latent_dim).to(device)
            fake_imgs = generator(z)
            
            # Loss for fake images
            fake_output = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            with torch.no_grad():
                real_accuracy = (real_output >= 0.5).float().mean().item()
                fake_accuracy = (fake_output < 0.5).float().mean().item()
                d_accuracy = 0.5 * (real_accuracy + fake_accuracy)

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate fake images and compute loss
            z = torch.randn(batch_size, args.latent_dim).to(device)
            gen_imgs = generator(z)
            output = discriminator(gen_imgs)
            g_loss = criterion(output, real_labels)
            
            g_loss.backward()
            optimizer_G.step()

            with torch.no_grad():
                g_accuracy = (output >= 0.5).float().mean().item()

            # Log losses
            batches_done = epoch * len(dataloader) + i
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            iterations.append(batches_done)
            d_real_acc.append(real_accuracy)
            d_fake_acc.append(fake_accuracy)
            d_avg_acc.append(d_accuracy)
            g_acc.append(g_accuracy)

            # Save Images
            # -----------
            if batches_done % args.save_interval == 0:
                save_image(gen_imgs[:25],
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True, value_range=(-1,1))
                print(f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                      f"[D acc: real {real_accuracy*100:.1f}% | fake {fake_accuracy*100:.1f}%] "
                      f"[G acc: {g_accuracy*100:.1f}%]")
    
    return (d_losses, g_losses, iterations,
            d_real_acc, d_fake_acc, d_avg_acc, g_acc,
            batches_per_epoch)


def plot_losses(d_losses, g_losses, iterations, save_path='gan_losses.png'):
    """Plot discriminator and generator losses."""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Both losses on same plot
    plt.subplot(1, 2, 1)
    plt.plot(iterations, d_losses, label='Discriminator Loss', alpha=0.7, linewidth=0.5)
    plt.plot(iterations, g_losses, label='Generator Loss', alpha=0.7, linewidth=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Smoothed losses (moving average)
    plt.subplot(1, 2, 2)
    window_size = 100
    if len(d_losses) >= window_size:
        d_losses_smooth = np.convolve(d_losses, np.ones(window_size)/window_size, mode='valid')
        g_losses_smooth = np.convolve(g_losses, np.ones(window_size)/window_size, mode='valid')
        iterations_smooth = iterations[window_size-1:]
        
        plt.plot(iterations_smooth, d_losses_smooth, label='Discriminator Loss (smoothed)', linewidth=2)
        plt.plot(iterations_smooth, g_losses_smooth, label='Generator Loss (smoothed)', linewidth=2)
    else:
        plt.plot(iterations, d_losses, label='Discriminator Loss', linewidth=2)
        plt.plot(iterations, g_losses, label='Generator Loss', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Loss plot saved to {save_path}")
    plt.show()


def plot_accuracy(d_real_acc, d_fake_acc, d_avg_acc, g_acc, iterations,
                  batches_per_epoch, save_path='gan_accuracy.png'):
    """Plot discriminator and generator accuracies with clearer summaries."""
    iterations = np.array(iterations)
    epochs = iterations // batches_per_epoch

    d_real_acc = np.array(d_real_acc) * 100
    d_fake_acc = np.array(d_fake_acc) * 100
    d_avg_acc = np.array(d_avg_acc) * 100
    g_acc = np.array(g_acc) * 100

    unique_epochs = np.unique(epochs)
    d_real_epoch = [d_real_acc[epochs == e].mean() for e in unique_epochs]
    d_fake_epoch = [d_fake_acc[epochs == e].mean() for e in unique_epochs]
    d_avg_epoch = [d_avg_acc[epochs == e].mean() for e in unique_epochs]
    g_epoch = [g_acc[epochs == e].mean() for e in unique_epochs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex='col')

    # Discriminator epoch averages
    ax = axes[0, 0]
    ax.plot(unique_epochs, d_real_epoch, label='Real accuracy (avg)', marker='o')
    ax.plot(unique_epochs, d_fake_epoch, label='Fake accuracy (avg)', marker='o')
    ax.plot(unique_epochs, d_avg_epoch, label='Avg accuracy', linewidth=2, color='tab:green')
    ax.axhline(50, color='grey', linestyle='--', linewidth=1, label='Chance level')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Discriminator Accuracy per Epoch')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Generator epoch averages
    ax = axes[0, 1]
    ax.plot(unique_epochs, g_epoch, label='Fooling rate (avg)', marker='o', color='tab:orange')
    ax.axhline(50, color='grey', linestyle='--', linewidth=1, label='Chance level')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Generator Fooling per Epoch')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Rolling discriminator accuracy (iteration scale)
    ax = axes[1, 0]
    window = max(200, batches_per_epoch // 2)
    kernel = np.ones(window) / window
    d_real_smooth = np.convolve(d_real_acc, kernel, mode='valid')
    d_fake_smooth = np.convolve(d_fake_acc, kernel, mode='valid')
    d_avg_smooth = np.convolve(d_avg_acc, kernel, mode='valid')
    it_smooth = iterations[window-1:]
    ax.plot(it_smooth, d_real_smooth, label='Real (smoothed)', linewidth=1)
    ax.plot(it_smooth, d_fake_smooth, label='Fake (smoothed)', linewidth=1)
    ax.fill_between(it_smooth, d_real_smooth, d_fake_smooth, color='tab:green', alpha=0.2,
                    label='Real/Fake gap')
    ax.plot(it_smooth, d_avg_smooth, label='Avg (smoothed)', color='tab:green', linewidth=2)
    ax.axhline(50, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'D accuracy (rolling mean, window={window})')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Rolling generator accuracy
    ax = axes[1, 1]
    g_smooth = np.convolve(g_acc, kernel, mode='valid')
    ax.plot(iterations, g_acc, color='tab:orange', alpha=0.2, label='Instantaneous')
    ax.plot(iterations[window-1:], g_smooth, color='tab:red', linewidth=2, label='Smoothed')
    ax.axhline(50, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Generator fooling (rolling mean, window={window})')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Accuracy plot saved to {save_path}")
    plt.show()


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Start training with logging
    print("Starting training...")
    (d_losses, g_losses, iterations,
     d_real_acc, d_fake_acc, d_avg_acc, g_acc,
     batches_per_epoch) = train_with_logging(
        dataloader, discriminator, generator, optimizer_G, optimizer_D, device, args
    )

    # Plot losses
    plot_losses(d_losses, g_losses, iterations, save_path='gan_losses.png')
    # Plot accuracies
    plot_accuracy(d_real_acc, d_fake_acc, d_avg_acc, g_acc, iterations,
                  batches_per_epoch, save_path='gan_accuracy.png')

    # Save generator for later use
    torch.save(generator.state_dict(), "mnist_generator.pt")
    print("Generator saved to mnist_generator.pt")
    
    # Save loss history
    np.savez('loss_history.npz', 
             d_losses=d_losses, 
             g_losses=g_losses, 
             iterations=iterations,
             d_real_acc=d_real_acc,
             d_fake_acc=d_fake_acc,
             d_avg_acc=d_avg_acc,
             g_acc=g_acc,
             batches_per_epoch=batches_per_epoch)
    print("Loss history saved to loss_history.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
