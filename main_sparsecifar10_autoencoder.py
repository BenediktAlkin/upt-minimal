from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm

from upt.models.approximator import Approximator
from upt.models.decoder_classifier import DecoderClassifier
from upt.models.encoder_supernodes import EncoderSupernodes
from upt.models.upt_sparseimage_classifier import UPTSparseImageClassifier
from upt.datasets.sparse_cifar10_autoencoder_dataset import SparseCifar10AutoencoderDataset
from upt.collators.sparseimage_classifier_collator import SparseImageClassifierCollator

def main():
    # initialize device
    device = torch.device("cuda")

    # initialize dataset
    data_root = Path("./data")
    data_root.mkdir(exist_ok=True)
    transform = ToTensor()
    train_dataset = SparseCifar10AutoencoderDataset(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
        # use half of the inputs for training (32x32 pixels = 1024)
        num_inputs=512,
        # use 3/4th of the outputs for training (32x32 pixels = 1024)
        num_outputs=768,
    )
    test_dataset = SparseCifar10AutoencoderDataset(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
        # use all inputs/outputs for evaluation (32x32 pixels = 1024)
        num_inputs=1024,
        num_outputs=1024,
    )

    # hyperparameters
    dim = 192  # ~6M parameter model
    num_heads = 3
    epochs = 10
    batch_size = 256

    # initialize model
    model = UPTSparseImageClassifier(
        encoder=EncoderSupernodes(
            # CIFAR has 3 channels (RGB)
            input_dim=3,
            # CIFAR is an image dataset -> 2D
            ndim=2,
            # there are 32x32 pixels so positions are in [0, 31], to have roughly the same input as a ViT
            # with patch_size=4, we'll use radius slighly larger than 4
            radius=5,
            # if we split a 32x32 image into 8x8 gridpoints, each point would cover 4x4 pixels, i.e. 16 pixels (=nodes)
            # since we sample supernodes randomly and use a larger radius, it can happen that more than 16 nodes
            # are in the radius of a supernode, so we'll use at maximum 32 connections to each supernode
            max_degree=32,
            # dimension for the supernode pooling -> use same as ViT-T latent dim
            gnn_dim=dim,
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
        decoder=DecoderClassifier(
            # tell the decoder the dimension of the input (dim of approximator)
            input_dim=dim,
            # CIFAR10 has 10 classes
            num_classes=10,
            # as in ViT-T
            dim=dim,
            num_heads=num_heads,
            # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
            depth=4,
        ),
    )
    model = model.to(device)
    print(model)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # setup dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=SparseImageClassifierCollator(num_supernodes=64, deterministic=False),
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=SparseImageClassifierCollator(num_supernodes=64, deterministic=True),
    )

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

    # train model
    update = 0
    pbar = tqdm(total=total_updates)
    pbar.update(0)
    pbar.set_description("train_loss: ????? train_accuracy: ????% test_accuracy: ????%")
    test_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    for _ in range(epochs):
        # train for an epoch
        model.train()
        for batch in train_dataloader:
            # schedule learning rate
            for param_group in optim.param_groups:
                param_group["lr"] = lrs[update]

            # forward pass
            y_hat = model(
                input_feat=batch["input_feat"].to(device),
                input_pos=batch["input_pos"].to(device),
                supernode_idxs=batch["supernode_idxs"].to(device),
                batch_idx=batch["batch_idx"].to(device),
            )
            y = batch["target_class"].to(device)
            loss = F.cross_entropy(y_hat, y)

            # backward pass
            loss.backward()

            # update step
            optim.step()
            optim.zero_grad()

            # status update
            train_accuracy = (y_hat.argmax(dim=1) == y).sum() / y.numel()
            update += 1
            pbar.update()
            pbar.set_description(
                f"train_loss: {loss.item():.4f} "
                f"train_accuracy: {train_accuracy * 100:4.1f}% "
                f"test_accuracy: {test_accuracy * 100:4.1f}%"
            )
            train_losses.append(loss.item())
            train_accuracies.append(train_accuracy)

        # evaluate
        num_correct = 0
        for batch in test_dataloader:
            y_hat = model(
                input_feat=batch["input_feat"].to(device),
                input_pos=batch["input_pos"].to(device),
                supernode_idxs=batch["supernode_idxs"].to(device),
                batch_idx=batch["batch_idx"].to(device),
            )
            y = batch["target_class"].to(device)
            num_correct += (y_hat.argmax(dim=1) == y).sum().item()
        test_accuracy = num_correct / len(test_dataset)
        test_accuracies.append(test_accuracy)
        pbar.set_description(
            f"train_loss: {loss.item():.4f} "
            f"train_accuracy: {train_accuracy * 100:4.1f}% "
            f"test_accuracy: {test_accuracy * 100:4.1f}%"
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(range(len(train_losses)), train_losses)
    axes[0].set_xlabel("Updates")
    axes[0].set_ylabel("Train Loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(range(len(train_accuracies)), train_accuracies)
    axes[1].set_xlabel("Updates")
    axes[1].set_ylabel("Train Accuracy")
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(range(len(test_accuracies)), test_accuracies, marker="o")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Test Accuracy")
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
