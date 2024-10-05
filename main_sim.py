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
from upt.models.upt import UPT

def main():
    # initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        mode="rollout",
    )

    # hyperparameters
    dim = 192
    num_heads = 3
    epochs = 1000
    batch_size = 19

    # initialize model
    conditioner = ConditionerTimestep(
        dim=dim,
        num_timesteps=train_dataset.num_timesteps,
    )
    model = UPT(
        conditioner=conditioner,
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
            # pass conditioner dim
            cond_dim=conditioner.cond_dim,
        ),
        approximator=Approximator(
            # tell the approximator the dimension of the input (perc_dim or enc_dim of encoder)
            input_dim=dim,
            # as in ViT-T
            dim=dim,
            num_heads=num_heads,
            # ViT-T has 12 blocks -> parameters are split evenly among encoder/approximator/decoder
            depth=4,
            # pass conditioner dim
            cond_dim=conditioner.cond_dim,
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
            # pass conditioner dim
            cond_dim=conditioner.cond_dim,
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
        batch_size=1,
        shuffle=False,
        drop_last=False,
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
    pbar.set_description("train_loss: ???????")
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
                timestep=batch["timestep"].to(device),
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
            pbar.set_description(f"train_loss: {loss.item():.6f}")
            train_losses.append(loss.item())


    test_batch = next(iter(rollout_dataloader))
    rollout_preds = model.rollout(
        input_feat=test_batch["input_feat"].to(device),
        input_pos=test_batch["input_pos"].to(device),
        supernode_idxs=test_batch["supernode_idxs"].to(device),
        batch_idx=test_batch["batch_idx"].to(device),
    )
    rollout_preds = [rollout_preds[i].cpu() for i in range(len(rollout_preds))]

    from pathlib import Path
    from matplotlib import patches
    out = Path("tmp")
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir()
    assert len(test_batch["output_feat"]) == 1, "batch_size > 1 not supported for rollout"
    output_feat = test_batch["output_feat"][0]
    num_rollout_timesteps = len(output_feat)


    import os
    if os.name == "nt":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    x, y = test_batch["input_pos"].unbind(1)
    for i in tqdm(range(num_rollout_timesteps)):
        plt.clf()
        fig = plt.figure(figsize=(30, 10))
        # needed to force correct aspect ratio
        positions = [
            [0.05, 0.1, 0.3, 0.8],
            [0.4, 0.1, 0.3, 0.8],
            [0.7, 0.1, 0.3, 0.8],
        ]
        ax = [fig.add_axes(pos) for pos in positions]

        # plot only velocity magnitude
        pred = rollout_preds[i][:, 1:]
        target = output_feat[i][:, 1:]
        delta = (pred - target).norm(dim=1)
        pred = pred.norm(dim=1)
        target = target.norm(dim=1)


        # format
        for ii in range(3):
            rect = patches.Rectangle((0, 0), 200, 300, facecolor="#ee8866", zorder=-10)
            ax[ii].add_patch(rect)
            ax[ii].set_axis_off()
            ax[ii].set_xlim(5, 295)
            ax[ii].set_ylim(5, 195)
            ax[ii].set_aspect(1.0)
        # plot displacement (plot with large pointsize first to fill whole area, then with small one for details)
        for point_size in [20, 4]:
            ax[0].scatter(x, y, c=target, s=point_size, cmap="bone", vmin=0, vmax=2.5)
            scatter1 = ax[1].scatter(x, y, c=pred, s=point_size, cmap="bone", vmin=0, vmax=2.5)
            scatter2 = ax[2].scatter(x, y, c=delta, s=point_size, cmap="bone", vmin=0, vmax=0.4)
        ax[0].set_title("target")
        ax[1].set_title("prediction")
        ax[2].set_title("delta")
        plt.colorbar(scatter1, ax=[ax[0], ax[1]], orientation="vertical")
        plt.colorbar(scatter2, ax=ax[2], orientation="vertical")
        # save
        plt.savefig(out / f"{i:04d}.png")
        plt.close()

    # create gif
    import imageio
    images = [
        imageio.imread(out / f"{i:04d}.png")
        for i in range(num_rollout_timesteps)
    ]
    imageio.mimsave(out / "rollout.gif", images, duration=0.1)



if __name__ == "__main__":
    main()
