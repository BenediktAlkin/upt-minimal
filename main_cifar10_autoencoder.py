from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from tqdm import tqdm

from upt.models.approximator import Approximator
from upt.models.decoder_perceiver import DecoderPerceiver
from upt.models.encoder_image import EncoderImage
from upt.models.upt_image_autoencoder import UPTImageAutoencoder


def main():
    # initialize device
    device = torch.device("cuda")

    # initialize dataset
    data_root = Path("./data")
    data_root.mkdir(exist_ok=True)
    transform = ToTensor()
    train_dataset = CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=data_root, train=False, download=True, transform=transform)

    # hyperparameters
    dim = 192  # ~6M parameter model
    num_heads = 3
    epochs = 10
    batch_size = 256

    # initialize model
    model = UPTImageAutoencoder(
        encoder=EncoderImage(
            # CIFAR has 3 channels (RGB)
            input_dim=3,
            # CIFAR has 32x32 images -> patch_size=4 results in 64 patch tokens
            resolution=32,
            patch_size=4,
            # ViT-T latent dimension
            enc_dim=dim,
            enc_num_heads=num_heads,
            # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
            enc_depth=4,
            # the perceiver is optional, it changes the size of the latent space to NUM_LATENT_TOKENS tokens
            # perc_dim=dim,
            # perc_num_heads=num_heads,
            # num_latent_tokens=32,
        ),
        approximator=Approximator(
            # tell the approximator the dimension of the input (perc_dim or enc_dim of encoder)
            input_dim=dim,
            # as in ViT-T
            dim=dim,
            num_heads=num_heads,
            # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
            depth=4,
        ),
        decoder=DecoderPerceiver(
            # tell the decoder the dimension of the input (dim of approximator)
            input_dim=dim,
            # images have 2D coordinates
            ndim=2,
            # as in ViT-T
            dim=dim,
            num_heads=num_heads,
            # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
            depth=4,
            # reshape to image after decoding
            unbatch_mode="image",
        ),
    )
    model = model.to(device)
    print(model)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # setup dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # initialize optimizer and learning rate schedule (linear warmup for first 10% -> linear decay)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    total_updates = len(train_dataloader) * epochs
    warmup_updates = int(total_updates * 0.1)
    lrs = torch.concat(
        [
            # linear warmup
            torch.linspace(0, optim.defaults["lr"], warmup_updates),
            # linear decay
            torch.linspace(optim.defaults["lr"], 0, total_updates - warmup_updates),
        ],
    )

    # output positions are fixed for training, we query on a regular grid
    # CIFAR has 32x32 pixels
    # output_pos will be a tensor of shape (32 * 32, 2) with and will contain x and y indices
    # output_pos[0] = [0, 0]
    # output_pos[1] = [0, 1]
    # output_pos[2] = [0, 2]
    # ...
    # output_pos[32] = [1, 0]
    # output_pos[1024] = [31, 31]
    output_pos = einops.rearrange(
        torch.stack(torch.meshgrid([torch.arange(32), torch.arange(32)], indexing="ij")),
        "ndim height width -> (height width) ndim",
    )
    output_pos = output_pos.to(device)
    # convert output_pos from [0, 31] to [0, 1000] for better behavior with sin-cos pos embeddings
    output_pos = output_pos / 31 * 1000
    # decoder needs float dtype
    output_pos = output_pos.float()

    # train model
    update = 0
    pbar = tqdm(total=total_updates)
    pbar.update(0)
    pbar.set_description("train_loss: ????? test_loss. ?????")
    train_losses = []
    test_losses = []
    test_loss = float("inf")
    for _ in range(epochs):
        # train for an epoch
        model.train()
        for x, _ in train_dataloader:
            # prepare forward pass
            x = x.to(device)

            # schedule learning rate
            for param_group in optim.param_groups:
                param_group["lr"] = lrs[update]

            # forward pass
            x_hat = model(x, output_pos=einops.repeat(output_pos, "... -> bs ...", bs=len(x)))
            loss = F.mse_loss(x_hat, x)

            # backward pass
            loss.backward()

            # update step
            optim.step()
            optim.zero_grad()

            # status update
            update += 1
            pbar.update()
            pbar.set_description(
                f"train_loss: {loss.item():.4f} "
                f"test_loss: {test_loss:.4f} "
            )
            train_losses.append(loss.item())

        # evaluate
        test_loss = 0.
        for x, _ in test_dataloader:
            x = x.to(device)
            x_hat = model(x)
            test_loss += F.mse_loss(x_hat, x, reduction="none").flatten(start_dim=1).mean(dim=1).sum().item()
        test_loss /= len(test_dataset)
        test_losses.append(test_loss)
        pbar.set_description(
            f"train_loss: {loss.item():.4f} "
            f"test_loss: {test_loss:.4f} "
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(range(len(train_losses)), train_losses)
    axes[0].set_xlabel("Updates")
    axes[0].set_ylabel("Train Loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(range(len(test_losses)), test_losses)
    axes[1].set_xlabel("Updates")
    axes[1].set_ylabel("Test Loss")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
