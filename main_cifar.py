import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from upt.models.approximator import Approximator
from upt.models.decoder_classifier import DecoderClassifier
from upt.models.encoder_image import EncoderImage
from upt.models.upt_classifier import UPTClassifier
from pathlib import Path

def main():
    # initialize device
    device = torch.device("cpu")

    # initialize dataset
    data_root = Path("./data")
    data_root.mkdir(exist_ok=True)
    transform = Compose([ToTensor(), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    train_dataset = CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=data_root, train=False, download=True, transform=transform)

    # hyperparameters
    dim = 6  # ~6M parameter model
    num_heads = 3
    epochs = 100
    batch_size = 256

    # initialize model
    model = UPTClassifier(
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

    # train model
    update = 0
    pbar = tqdm(total=total_updates)
    pbar.update(0)
    pbar.set_description("train loss: ????? train accuracy: ????% test_accuracy: ????%")
    test_accuracy = 0.0
    for _ in range(epochs):
        # train for an epoch
        model.train()
        for x, y in train_dataloader:
            # prepare forward pass
            x = x.to(device)
            y = y.to(device)

            # schedule learning rate
            for param_group in optim.param_groups:
                param_group["lr"] = lrs[update]

            # forward pass
            y_hat = model(x)
            loss = F.cross_entropy(y_hat, y)

            # backward pass
            loss.backward()

            # update step
            optim.step()
            optim.zero_grad()

            # status update
            train_accuracy = (y_hat.argmax(dim=1) == y).sum() / y.numel()
            update += 1
            pbar.update(update)
            pbar.set_description(
                f"train loss: {loss.item():.4f} "
                f"train accuracy: {train_accuracy * 100:4.1f}% "
                f"test_accuracy: {test_accuracy * 100:4.1f}%"
            )


        # evaluate
        num_correct = 0
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            num_correct += (y_hat.argmax(dim=1) == y).sum().item()
        test_accuracy = num_correct / len(test_dataset)
        pbar.set_description(
            f"train loss: {loss.item():.4f} "
            f"train accuracy: {train_accuracy * 100:4.1f}% "
            f"test_accuracy: {test_accuracy * 100:4.1f}%"
        )

if __name__ == "__main__":
    main()
