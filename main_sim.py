import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from upt.collators.simulation_collator import SimulationCollator
from upt.datasets.simulation_dataset import SimulationDataset
from upt.models.approximator import Approximator
from upt.models.decoder_perceiver import DecoderPerceiver
from upt.models.encoder_supernodes import EncoderSupernodes
from upt.models.conditioner_timestep import ConditionerTimestep


def main():
    # initialize device
    device = torch.device("cuda")

    # initialize dataset
    train_dataset = SimulationDataset(
        root="./data/simulation",
        # how many inputs to use for training
        num_inputs=8192,
        # how many outputs to use for training
        num_outputs=4096,
        # mode
        mode="train",
    )
    rollout_dataset = SimulationDataset(
        root="./data/simulation",
        # use all inputs for rollout
        num_inputs=float("inf"),
        # use all outputs for rollout
        num_outputs=float("inf"),
        # mode
        mode="train",
    )

    # hyperparameters
    dim = 192  # ~6M parameter model
    num_heads = 3
    epochs = 10
    batch_size = 256

    # initialize model
    model = UPT(
        conditioner=ConditionerTimestep(
            dim=dim,
            num_timesteps=train_dataset.num_timesteps,
        ),
        encoder=EncoderSupernodes(
            # simulation has 3 inputs: 2D velocity + pressure
            input_dim=3,
            # 2D dataset
            ndim=2,
            # positions are rescaled to [0, 200]
            radius=5,
            # in regions where there are a lot of mesh cells, it would result in supernodes having a lot of
            # connections to nodes. but since we sample the supernodes uniform, we also have a lot of supernodes
            # in dense regions, so we can simply limit the maximum amount of connections to each supernodes
            # to avoid an extreme amount of edges
            max_degree=32,
            # dimension for the supernode pooling -> use same as ViT-T latent dim
            gnn_dim=dim,
            # ViT-T latent dimension
            enc_dim=dim,
            enc_num_heads=num_heads,
            # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
            enc_depth=4,
            # downsample to 128 latent tokens for fast training
            perc_dim=dim,
            perc_num_heads=num_heads,
            num_latent_tokens=128,
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
            # 2D velocity + pressure
            output_dim=3,
            # simulation is 2D
            ndim=2,
            # as in ViT-T
            dim=dim,
            num_heads=num_heads,
            # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
            depth=4,
            # we assume num_outputs to be constant so we can simply reshape the dense result into a sparse tensor
            unbatch_mode="dense_to_sparse_unpadded",
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
        collate_fn=SimulationCollator(num_supernodes=512, deterministic=False),
    )
    rollout_dataloader = DataLoader(
        dataset=rollout_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=SimulationCollator(num_supernodes=512, deterministic=True),
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
    pbar.set_description("train_loss: ?????")
    train_losses = []
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
                output_pos=batch["output_pos"].to(device),
            )
            y = batch["output_feat"].to(device)
            loss = F.mse_loss(y_hat, y)

            # backward pass
            loss.backward()

            # update step
            optim.step()
            optim.zero_grad()

            # status update
            update += 1
            pbar.update()
            pbar.set_description(f"train_loss: {loss.item():.4f}")
            train_losses.append(loss.item())




if __name__ == "__main__":
    main()
